"""
Custom feature augmentation for the «акылай» wake word — replaces the built-in
`train.py --augment_clips` step so we can control augmentation probabilities AND add
the device-specific colorations found in the 20-clip analysis (device_analysis/).

Recipe (BALANCED, parametric), applied to BOTH positives and negatives, order:
    device-color (low-pass + mid-boost)  ->  RIR колонки  ->  фон+MUSAN  ->  pitch  ->  AGC

Why each (see device_analysis/stats.txt):
  device-color : device rolloff 1548 Hz vs train 2567 Hz, +5..13 dB hump at 2-3.5 kHz,
                 -10..15 dB above 4 kHz  -> low-pass + mid peak EQ shifts a train clip
                 toward the device timbre.
  RIR          : device flatness ~2.5x lower (resonance) -> convolve with speaker IR.
  фон+MUSAN    : robustness to real noise.
  pitch        : speaker variety.
  AGC          : ch0 is AEC+AGC processed (slight attenuation, duller highs).

Output (same files train.py --train_model reads, so run WITHOUT --augment_clips afterwards):
    oww/akylai/positive_features_train.npy
    oww/akylai/negative_features_train.npy
    oww/akylai/positive_features_test.npy
    oww/akylai/negative_features_test.npy

Run from repo root:
    python training/akylai/augment_features.py            # full
    python training/akylai/augment_features.py --subset 64  # quick dry-run
    python openwakeword/train.py --training_config training/akylai/akylai.yml --train_model
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch
import torchaudio
import torch_audiomentations
from scipy.signal import butter, sosfilt, lfilter
from speechbrain.processing.signal_processing import reverberate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root
import openwakeword.data
from openwakeword.data import create_fixed_size_clip
from openwakeword.utils import compute_features_from_generator
from numpy.lib.format import open_memmap


def _safe_trim_mmap(mmap_path):
    """Windows-safe replacement for openwakeword.data.trim_mmap: closes the memmap
    handle before os.remove/rename (numpy memmap otherwise keeps the file locked)."""
    import gc
    m1 = np.load(mmap_path, mmap_mode="r")
    i = -1
    while np.all(m1[i, :, :] == 0):
        i -= 1
    n_new = m1.shape[0] + i + 1
    if n_new >= m1.shape[0]:                     # nothing to trim
        m1._mmap.close(); del m1; gc.collect()
        return
    tmp = mmap_path[:-4] + "_trim.npy"
    m2 = open_memmap(tmp, mode="w+", dtype=np.float32,
                     shape=(n_new, m1.shape[1], m1.shape[2]))
    for j in range(0, n_new, 1024):
        end = min(j + 1024, n_new)
        m2[j:end] = m1[j:end]
    m2.flush()
    m1._mmap.close(); del m1
    m2._mmap.close(); del m2
    gc.collect()
    os.remove(mmap_path)
    os.rename(tmp, mmap_path)


openwakeword.data.trim_mmap = _safe_trim_mmap   # patch (compute_features imports it at call time)

# ── paths ───────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
WORK        = Path(r"D:/akylai_oww")    # data + outputs on D: (C: is nearly full)
OWW         = WORK / "oww" / "akylai"
FEATURE_DIR = OWW
BACKGROUND_DIRS = [WORK / "oww" / "backgrounds_wav", WORK / "oww" / "musan_wav"]
RIR_DIR     = WORK / "oww" / "rirs_device"
SR          = 16000
SEED        = 42

# ══ PARAMS — CONSERVATIVE recipe (gentler; test if Balanced was too harsh) ════════
# Balanced was: device_color .60, lp 3-4k, mid +5..9, rir .70, bg .75 SNR -5..15, agc .30
PARAMS = dict(
    # device-color (per clip): low-pass + mid peak-EQ
    p_device_color = 0.40,
    lp_cutoff_hz   = (3500, 4500),   # gentler low-pass (keep more highs)
    lp_order       = (2, 4),
    mid_center_hz  = (2000, 3500),
    mid_gain_db    = (3.0, 6.0),     # gentler boost
    mid_q          = (0.7, 1.5),

    # RIR колонки (per batch)
    p_rir = 0.40,

    # background + MUSAN (per batch) — SNR matches dataset designers (5..25), not -5
    p_background = 0.60,
    snr_db       = (5.0, 20.0),

    # pitch (per batch)
    p_pitch       = 0.25,
    pitch_semitones = (-3, 3),

    # AGC/AEC coloring — OFF in conservative run
    p_agc      = 0.0,
    agc_att_db = (5.0, 8.0),
    agc_lp_hz  = (4500, 6000),
)
# ═════════════════════════════════════════════════════════════════════════════════


# ── per-clip filters ─────────────────────────────────────────────────────────────
def lowpass(x, cutoff, order, fs=SR):
    sos = butter(order, cutoff, btype="low", fs=fs, output="sos")
    return sosfilt(sos, x)


def peaking_eq(x, f0, gain_db, q, fs=SR):
    """RBJ cookbook peaking EQ biquad."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * q)
    cosw = np.cos(w0)
    b = np.array([1 + alpha * A, -2 * cosw, 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * cosw, 1 - alpha / A])
    return lfilter(b / a[0], a / a[0], x)


def apply_device_color(x):
    x = lowpass(x, np.random.uniform(*PARAMS["lp_cutoff_hz"]),
                np.random.randint(PARAMS["lp_order"][0], PARAMS["lp_order"][1] + 1))
    x = peaking_eq(x, np.random.uniform(*PARAMS["mid_center_hz"]),
                   np.random.uniform(*PARAMS["mid_gain_db"]),
                   np.random.uniform(*PARAMS["mid_q"]))
    m = np.max(np.abs(x))
    if m > 1.0:
        x = x / m
    return x


def apply_agc(x):
    x = x * 10 ** (-np.random.uniform(*PARAMS["agc_att_db"]) / 20.0)
    x = lowpass(x, np.random.uniform(*PARAMS["agc_lp_hz"]), 2)
    return x


# ── augmentation generator (mirrors data.augment_clips contract) ─────────────────
def device_augment_generator(clip_paths, total_length, background_paths, rir_paths,
                             batch_size=16):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # batch transforms: background THEN pitch (recipe order)
    batch_aug = torch_audiomentations.Compose([
        torch_audiomentations.AddBackgroundNoise(
            p=PARAMS["p_background"], background_paths=background_paths,
            min_snr_in_db=PARAMS["snr_db"][0], max_snr_in_db=PARAMS["snr_db"][1],
            mode="per_batch"),
        torch_audiomentations.PitchShift(
            p=PARAMS["p_pitch"], sample_rate=SR, mode="per_batch",
            min_transpose_semitones=PARAMS["pitch_semitones"][0],
            max_transpose_semitones=PARAMS["pitch_semitones"][1]),
    ])

    for i in range(0, len(clip_paths), batch_size):
        batch = clip_paths[i:i + batch_size]
        clips = []
        for clip in batch:
            try:
                data, csr = torchaudio.load(clip)
            except Exception:
                continue
            if csr != SR:
                raise ValueError(f"{clip}: expected {SR} Hz, got {csr}")
            data = data[0]
            if data.shape[0] > total_length:
                data = data[0:total_length]
            x = create_fixed_size_clip(data, total_length, SR)          # left-pad to fixed size
            if np.random.random() < PARAMS["p_device_color"]:           # 1) device-color
                x = apply_device_color(x)
            clips.append(torch.from_numpy(np.asarray(x, dtype=np.float32)))

        if not clips:
            continue
        batch_t = torch.vstack(clips)                                    # (B, T)

        # 2) RIR колонки (whole batch)
        if rir_paths and np.random.random() <= PARAMS["p_rir"]:
            rir, _ = torchaudio.load(random.choice(rir_paths))
            if rir.shape[0] > 1:
                rir = rir[random.randint(0, rir.shape[0] - 1), :]
            batch_t = reverberate(batch_t, rir, rescale_amp="avg")

        # 3+4) background+MUSAN, then pitch
        batch_t = batch_aug(samples=batch_t.unsqueeze(1).to(device),
                            sample_rate=SR).squeeze(1).cpu()

        # 5) AGC/AEC coloring (per clip)
        out = batch_t.numpy()
        for j in range(out.shape[0]):
            if np.random.random() < PARAMS["p_agc"]:
                out[j] = apply_agc(out[j])

        out = np.clip(out, -1.0, 1.0)
        yield (out * 32767).astype(np.int16)


# ── total_length, same logic as train.py ─────────────────────────────────────────
def compute_total_length(positive_test_dir, n=50):
    clips = [str(p) for p in Path(positive_test_dir).glob("*.wav")]
    durs = []
    for _ in range(min(n, len(clips))):
        _, dat = scipy.io.wavfile.read(clips[np.random.randint(0, len(clips))])
        durs.append(len(dat))
    total = int(round(np.median(durs) / 1000) * 1000) + 12000
    if total < 32000:
        total = 32000
    elif abs(total - 32000) <= 4000:
        total = 32000
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=0,
                    help="limit clips per category for a quick dry-run (0 = all)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--train-only", action="store_true",
                    help="regenerate only train feature files; keep existing test .npy")
    args = ap.parse_args()

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    background_paths = [str(p) for d in BACKGROUND_DIRS for p in Path(d).glob("*.wav")]
    rir_paths = [str(p) for p in Path(RIR_DIR).glob("*.wav")]
    print(f"backgrounds: {len(background_paths)}   RIRs: {len(rir_paths)}")

    total_length = compute_total_length(OWW / "positive_test")
    print(f"total_length: {total_length} samples ({total_length/SR:.2f} s)")

    sets = [
        ("positive_train", "positive_features_train.npy"),
        ("negative_train", "negative_features_train.npy"),
        ("positive_test",  "positive_features_test.npy"),
        ("negative_test",  "negative_features_test.npy"),
    ]
    if args.train_only:
        sets = sets[:2]
        print("--train-only: keeping existing test feature files")

    n_cpus = max(1, (os.cpu_count() or 2) // 2)

    for clip_dir, out_name in sets:
        files = [str(p) for p in Path(OWW / clip_dir).glob("*.wav")]
        files.sort()
        if args.subset:
            files = files[:args.subset]
        n_total = len(files)
        out_file = str(FEATURE_DIR / out_name)
        print(f"\n=== {clip_dir}: {n_total} clips -> {out_name} ===")
        gen = device_augment_generator(files, total_length, background_paths,
                                       rir_paths, batch_size=args.batch_size)
        compute_features_from_generator(
            gen, n_total=n_total, clip_duration=total_length, output_file=out_file,
            device="gpu" if torch.cuda.is_available() else "cpu", ncpu=n_cpus)
        shp = np.load(out_file, mmap_mode="r").shape
        print(f"  saved {out_name}  shape={shp}")

    print("\nDone. Next: python openwakeword/train.py "
          "--training_config training/akylai/akylai.yml --train_model")


if __name__ == "__main__":
    main()
