"""
Acoustic analysis of the 20 real-device captures (6-mic array) vs the openWakeWord
training positives.

Device clips: training/akylai/device_clips/clip_*.wav  (6 ch, 16 kHz, 5 s)
  ch0      = processed / beamformed (what the wake-word model receives on-device)  <-- primary
  ch1..4   = raw mic capsules
  ch5      = empty playback/loopback reference

Train corpus: training/akylai/oww/akylai/positive_train/*.wav  (mono 16 kHz, isolated "akylai")

Outputs (training/akylai/device_analysis/):
  stats.txt            numeric summary + JSD vs train + notes
  segments.txt         detected speech segments per device clip (ch0)
  fig1_overview.png    device waveforms+VAD and mel spectrograms (does it contain the word?)
  fig2_features.png    feature distributions device(ch0) vs train + JSD
  fig3_spectral.png    mean spectral envelope + gap, mel-band delta
"""
from __future__ import annotations

import random
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.spatial.distance import jensenshannon

HERE       = Path(__file__).parent
DEVICE_DIR = HERE / "device_clips"
TRAIN_DIR  = HERE / "oww" / "akylai" / "positive_train"
OUT_DIR    = HERE / "device_analysis"
OUT_DIR.mkdir(exist_ok=True)

SR      = 16000
N_FFT   = 512
HOP     = 160          # 10 ms
N_MELS  = 64
N_MFCC  = 20
VAD_DB  = -38.0
N_TRAIN = 60
SEED    = 42

random.seed(SEED)
np.random.seed(SEED)

freqs_hz  = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=SR / 2)


def load_ch0(path: Path) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = a[:, 0]                       # processed channel
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x


def load_mono(path: Path) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    return a


def normalize(a: np.ndarray) -> np.ndarray:
    return a / (np.max(np.abs(a)) + 1e-9)


def rms_db(a: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-9)


def detect_segments(audio, thresh_db=VAD_DB, frame_len=400, hop_len=160,
                    min_frames=15, pad_frames=8):
    frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop_len)
    nrg = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=0)) + 1e-9)
    mask = nrg > thresh_db
    padded = np.pad(mask, pad_frames, constant_values=False)
    padded = np.convolve(padded.astype(float), np.ones(pad_frames * 2 + 1), "same") > 0
    mask = padded[pad_frames:-pad_frames]
    segs, in_seg, start = [], False, 0
    for i, v in enumerate(mask):
        if v and not in_seg:
            in_seg, start = True, i
        elif not v and in_seg:
            in_seg = False
            if i - start >= min_frames:
                segs.append((start * hop_len / SR, i * hop_len / SR))
    if in_seg and len(mask) - start >= min_frames:
        segs.append((start * hop_len / SR, len(audio) / SR))
    return segs, nrg


def spectral_envelope(audio):
    S = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP))
    return librosa.amplitude_to_db(S.mean(axis=1))


def features(clips):
    out = dict(centroid=[], rolloff=[], zcr=[], flatness=[], rms_db=[],
               mfcc=[], mel_mean=[])
    for c in clips:
        c = normalize(c)
        S = np.abs(librosa.stft(c, n_fft=N_FFT, hop_length=HOP))
        mel = librosa.feature.melspectrogram(S=S ** 2, sr=SR, n_mels=N_MELS,
                                             n_fft=N_FFT, hop_length=HOP)
        mel_db = librosa.power_to_db(mel)
        out["mfcc"].append(librosa.feature.mfcc(S=mel_db, n_mfcc=N_MFCC).mean(1))
        out["centroid"].append(librosa.feature.spectral_centroid(S=S, sr=SR).mean())
        out["rolloff"].append(librosa.feature.spectral_rolloff(S=S, sr=SR).mean())
        out["zcr"].append(librosa.feature.zero_crossing_rate(c).mean())
        out["flatness"].append(librosa.feature.spectral_flatness(S=S).mean())
        out["rms_db"].append(rms_db(c))
        out["mel_mean"].append(mel_db.mean(1))
    return {k: np.array(v) for k, v in out.items()}


def hist_jsd(a, b, bins=40):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    pa, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    pb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    pa = pa + 1e-9; pa /= pa.sum()
    pb = pb + 1e-9; pb /= pb.sum()
    return float(jensenshannon(pa, pb))


# -- load device clips (ch0) and VAD --------------------------------------------
dev_files = sorted(DEVICE_DIR.glob("clip_*.wav"))
print(f"Device clips: {len(dev_files)}")
dev_audio, dev_segments, dev_clips = [], [], []
for p in dev_files:
    x = load_ch0(p)
    segs, _ = detect_segments(x)
    dev_audio.append(x)
    dev_segments.append(segs)
    if segs:
        t0, t1 = max(segs, key=lambda s: s[1] - s[0])   # longest segment = utterance
        a = max(0, int((t0 - 0.1) * SR)); b = min(len(x), int((t1 + 0.1) * SR))
        dev_clips.append(x[a:b])
    else:
        dev_clips.append(x)

# -- load training positives ----------------------------------------------------
train_files = sorted(TRAIN_DIR.glob("*.wav"))
train_files = random.sample(train_files, min(N_TRAIN, len(train_files)))
train_clips = [load_mono(p) for p in train_files]
print(f"Train clips: {len(train_clips)}")

dev_feat   = features(dev_clips)
train_feat = features(train_clips)
dev_env    = np.array([spectral_envelope(normalize(c)) for c in dev_clips])
train_env  = np.array([spectral_envelope(normalize(c)) for c in train_clips])

FEATS = ["centroid", "rolloff", "rms_db", "zcr", "flatness"]
FLABEL = {"centroid": "Spectral centroid (Hz)", "rolloff": "Spectral rolloff (Hz)",
          "rms_db": "RMS of normalized clip (dB)", "zcr": "Zero-crossing rate",
          "flatness": "Spectral flatness"}

# -- segments.txt ---------------------------------------------------------------
with open(OUT_DIR / "segments.txt", "w", encoding="utf-8") as f:
    for p, segs in zip(dev_files, dev_segments):
        f.write(f"\n# {p.name}  ({len(segs)} segs)\n")
        f.write(f"{'#':>3}  {'start':>7}  {'end':>7}  {'dur':>6}\n")
        for i, (t0, t1) in enumerate(segs):
            f.write(f"{i:>3}  {t0:>7.3f}  {t1:>7.3f}  {t1 - t0:>6.3f}\n")

# -- stats.txt ------------------------------------------------------------------
def fmt(feat, k):
    return f"{feat[k].mean():>8.2f} +/-{feat[k].std():.2f}"

lines = [
    "DEVICE CAPTURES (6-mic array, ch0 processed)  vs  openWakeWord TRAIN POSITIVES",
    "=" * 78,
    f"device clips: {len(dev_clips)}   train clips sampled: {len(train_clips)}",
    "",
    f"{'Source':<16}{'Centroid Hz':>18}{'Rolloff Hz':>18}{'RMSnorm dB':>16}{'ZCR':>12}{'Flatness':>14}",
    "-" * 94,
    f"{'TRAIN':<16}{fmt(train_feat,'centroid'):>18}{fmt(train_feat,'rolloff'):>18}"
    f"{fmt(train_feat,'rms_db'):>16}{fmt(train_feat,'zcr'):>12}{fmt(train_feat,'flatness'):>14}",
    f"{'DEVICE ch0':<16}{fmt(dev_feat,'centroid'):>18}{fmt(dev_feat,'rolloff'):>18}"
    f"{fmt(dev_feat,'rms_db'):>16}{fmt(dev_feat,'zcr'):>12}{fmt(dev_feat,'flatness'):>14}",
    "",
    "JSD device(ch0) vs train  (0 = identical, 1 = disjoint)",
    "-" * 94,
]
for fk in FEATS:
    lines.append(f"  {fk:<12} {hist_jsd(train_feat[fk], dev_feat[fk]):.3f}")
avg_jsd = np.mean([hist_jsd(train_feat[fk], dev_feat[fk]) for fk in FEATS])
lines += ["", f"  AVG JSD = {avg_jsd:.3f}", ""]

abs_rms = [rms_db(x) for x in dev_audio]
lines += [
    "DEVICE ABSOLUTE LEVEL (ch0, whole 5 s clip, before normalization)",
    "-" * 94,
    f"  mean RMS = {np.mean(abs_rms):.1f} dBFS   range [{min(abs_rms):.1f}, {max(abs_rms):.1f}]",
    "",
    "REFERENCE -- previous spectral_search device sessions (centroid Hz):",
    "  train~1298 | raw_mic~718 | aec~675 | crowd~929 | female~1539",
]
(OUT_DIR / "stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

# -- fig1: overview waveforms + mel spectrograms --------------------------------
n_show = 6
fig, axes = plt.subplots(n_show, 2, figsize=(14, 2.0 * n_show))
fig.suptitle("Device clips (ch0): waveform + VAD (left), mel spectrogram (right)",
             fontsize=12, fontweight="bold")
for r in range(n_show):
    x = dev_audio[r]; segs = dev_segments[r]
    ax = axes[r, 0]
    t = np.linspace(0, len(x) / SR, len(x))
    ax.plot(t, x, lw=0.4, color="gray")
    for t0, t1 in segs:
        ax.axvspan(t0, t1, alpha=0.3, color="#e05a2b")
    ax.set_title(f"{dev_files[r].name}  ({len(segs)} segs, RMS {rms_db(x):.1f} dBFS)",
                 fontsize=8)
    ax.tick_params(labelsize=6)
    ax = axes[r, 1]
    mel = librosa.feature.melspectrogram(y=normalize(x), sr=SR, n_mels=N_MELS,
                                         n_fft=N_FFT, hop_length=HOP)
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=SR,
                             hop_length=HOP, x_axis="time", y_axis="mel",
                             fmax=SR // 2, ax=ax)
    ax.set_title(dev_files[r].name, fontsize=8)
    ax.tick_params(labelsize=6)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(OUT_DIR / "fig1_overview.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# -- fig2: feature distributions ------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Feature distributions: DEVICE ch0 vs TRAIN", fontsize=12, fontweight="bold")
for ax, fk in zip(axes.flat, FEATS):
    a, b = train_feat[fk], dev_feat[fk]
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    kw = dict(bins=18, range=(lo, hi), density=True, alpha=0.55)
    ax.hist(a, color="#2b7be0", label="train", **kw)
    ax.hist(b, color="#e05a2b", label="device", **kw)
    ax.set_title(f"{FLABEL[fk]}\nJSD = {hist_jsd(a, b):.2f}", fontsize=9)
    ax.legend(fontsize=7); ax.tick_params(labelsize=7)
axes.flat[-1].axis("off")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_DIR / "fig2_features.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# -- fig3: spectral envelope + gap + mel-band delta -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Spectral domain gap: DEVICE ch0 vs TRAIN", fontsize=12, fontweight="bold")

ax = axes[0]
for e in train_env:
    ax.plot(freqs_hz, e, color="#2b7be0", alpha=0.05, lw=0.5)
for e in dev_env:
    ax.plot(freqs_hz, e, color="#e05a2b", alpha=0.12, lw=0.6)
ax.plot(freqs_hz, train_env.mean(0), color="#2b7be0", lw=2.5, label="train mean")
ax.plot(freqs_hz, dev_env.mean(0), color="#e05a2b", lw=2.5, label="device mean")
ax.set_xlabel("Hz"); ax.set_ylabel("dB"); ax.legend(fontsize=8)
ax.set_title("Mean log spectral envelope", fontsize=9)

ax = axes[1]
diff = dev_env.mean(0) - train_env.mean(0)
ax.plot(freqs_hz, diff, color="#8e44ad", lw=1.8)
ax.fill_between(freqs_hz, diff, 0, where=(diff < 0), alpha=0.2, color="#e05a2b",
                label="device weaker")
ax.fill_between(freqs_hz, diff, 0, where=(diff > 0), alpha=0.2, color="#27ae60",
                label="device stronger")
ax.axhline(0, color="black", lw=0.8, ls=":")
ax.set_xlabel("Hz"); ax.set_ylabel("dDB (device - train)"); ax.legend(fontsize=8)
ax.set_title("Spectral gap", fontsize=9)

ax = axes[2]
d = dev_feat["mel_mean"].mean(0) - train_feat["mel_mean"].mean(0)
ax.bar(np.arange(N_MELS), d,
       color=["#27ae60" if v > 0 else "#e05a2b" for v in d], alpha=0.8)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("mel band"); ax.set_ylabel("dDB (device - train)")
ax.set_title("Mel-band delta", fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_DIR / "fig3_spectral.png", dpi=110, bbox_inches="tight")
plt.close(fig)

print(f"\nDone -> {OUT_DIR}")
for fn in ["stats.txt", "segments.txt", "fig1_overview.png",
           "fig2_features.png", "fig3_spectral.png"]:
    print("   ", fn)
