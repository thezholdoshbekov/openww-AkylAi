"""Deep acoustic analysis of the AkylAi speaker-device audio.

Answers: is there reverberation (RT60), what is the device frequency coloration,
how noisy are the real sessions — to decide the device-RIR augmentation proportion.

Inputs (from the sibling livekit_ww_akylai project):
  spectral_search/rir_device.wav   measured device impulse response (16 kHz)
  spectral_search/recording.wav    raw sweep capture
  ww_session*.wav                  real wake-word sessions (quiet / crowded / female)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

LK = Path(r"C:/Users/THEZHOLDOSHBEKOV/livekit_ww_akylai")
SR = 16000
OUT = Path(__file__).resolve().parent / "analysis"
OUT.mkdir(exist_ok=True)

TRAIN_CENTROID_HZ = 1300.0  # reference from spectral_search/stats.txt


def load(path: Path) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        import librosa
        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    return a


def spectral_stats(x: np.ndarray) -> dict:
    """Frame-based spectral centroid / rolloff85 / flatness, plus RMS dBFS."""
    n_fft, hop = 512, 256
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    frames = [x[i:i + n_fft] * np.hanning(n_fft)
              for i in range(0, len(x) - n_fft, hop)]
    mags = np.abs(np.fft.rfft(np.array(frames), axis=1)) + 1e-10
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    centroid = (mags * freqs).sum(1) / mags.sum(1)
    cumsum = np.cumsum(mags, axis=1)
    rolloff = freqs[(cumsum >= 0.85 * cumsum[:, -1:]).argmax(1)]
    gmean = np.exp(np.log(mags).mean(1))
    flatness = gmean / mags.mean(1)
    rms = np.sqrt((np.array(frames) ** 2).mean(1)) + 1e-10
    return dict(
        centroid=float(np.median(centroid)),
        rolloff=float(np.median(rolloff)),
        flatness=float(np.median(flatness)),
        rms_db=20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-10),
        frame_rms_db=20 * np.log10(rms),
    )


def analyze_rir(h: np.ndarray) -> dict:
    """Energy decay (Schroeder), RT estimates, DRR, frequency response."""
    h = h / (np.max(np.abs(h)) + 1e-12)
    peak = int(np.argmax(np.abs(h)))
    h = h[peak:]                                   # align to direct sound
    energy = h ** 2

    # Direct-to-reverberant ratio: direct window = +/- 2.5 ms around the peak
    d = int(0.0025 * SR)
    direct = energy[:d].sum()
    revb = energy[d:].sum() + 1e-12
    drr_db = 10 * np.log10(direct / revb)

    # Schroeder backward integration -> EDC in dB
    edc = np.cumsum(energy[::-1])[::-1]
    edc_db = 10 * np.log10(edc / edc[0] + 1e-12)
    t = np.arange(len(edc_db)) / SR

    def rt_from(lo, hi):
        i0 = np.argmax(edc_db <= lo)
        i1 = np.argmax(edc_db <= hi)
        if i1 <= i0:
            return None
        slope = (edc_db[i1] - edc_db[i0]) / (t[i1] - t[i0])  # dB/s
        return -60.0 / slope

    rt20 = rt_from(-5, -25)   # T20 -> RT60
    rt30 = rt_from(-5, -35)   # T30 -> RT60 (may be None if RIR too short)

    # time to decay 60 dB within the measured window (or cap)
    i60 = np.argmax(edc_db <= -60)
    decay60 = t[i60] if edc_db.min() <= -60 else None

    # Frequency response of the device IR
    H = np.abs(np.fft.rfft(h)) + 1e-10
    f = np.fft.rfftfreq(len(h), 1 / SR)
    Hdb = 20 * np.log10(H / H.max())
    centroid = float((H * f).sum() / H.sum())

    def band_gain(f1, f2):
        m = (f >= f1) & (f < f2)
        return float(Hdb[m].mean()) if m.any() else float("nan")

    bands = {"0-250": band_gain(0, 250), "250-500": band_gain(250, 500),
             "500-1k": band_gain(500, 1000), "1k-2k": band_gain(1000, 2000),
             "2k-4k": band_gain(2000, 4000), "4k-8k": band_gain(4000, 8000)}
    # -3 dB / -10 dB cutoff (low-pass corner)
    above = f[(Hdb <= -10) & (f > centroid)]
    lp10 = float(above[0]) if len(above) else None
    return dict(len_ms=len(h) / SR * 1000, drr_db=drr_db, rt20=rt20, rt30=rt30,
                decay60=decay60, ir_centroid=centroid, lp10_hz=lp10, bands=bands,
                edc_db=edc_db, t=t, f=f, Hdb=Hdb)


def reverb_proxy(x: np.ndarray) -> dict:
    """Blind reverb/noise proxies on a real recording.
    - dyn_range_db: 95th-5th pct of frame energy (low -> reverb/noise fills pauses)
    - noise_floor_db / speech_db / est_snr_db
    """
    n_fft, hop = 512, 256
    frames = np.array([x[i:i + n_fft] for i in range(0, len(x) - n_fft, hop)])
    fr_db = 10 * np.log10((frames ** 2).mean(1) + 1e-12)
    p5, p50, p95 = np.percentile(fr_db, [5, 50, 95])
    return dict(dyn_range_db=p95 - p5, noise_floor_db=p5, speech_db=p95,
                est_snr_db=p95 - p5)


def main() -> None:
    print("=" * 78)
    print("DEVICE IMPULSE RESPONSE — rir_device.wav")
    print("=" * 78)
    h = load(LK / "spectral_search" / "rir_device.wav")
    r = analyze_rir(h)
    print(f"  length window : {r['len_ms']:.0f} ms")
    print(f"  DRR (direct/reverb) : {r['drr_db']:+.1f} dB   "
          f"({'reverb negligible' if r['drr_db'] > 10 else 'reverb present'})")
    print(f"  RT60 (from T20) : {('%.0f ms' % (r['rt20']*1000)) if r['rt20'] else 'n/a'}")
    print(f"  RT60 (from T30) : {('%.0f ms' % (r['rt30']*1000)) if r['rt30'] else 'n/a (IR shorter than 35 dB decay)'}")
    print(f"  full -60 dB decay reached at : {('%.0f ms' % (r['decay60']*1000)) if r['decay60'] else 'not within window'}")
    print(f"  IR spectral centroid : {r['ir_centroid']:.0f} Hz   (train corpus ~{TRAIN_CENTROID_HZ:.0f} Hz)")
    print(f"  low-pass corner (-10 dB) : {('%.0f Hz' % r['lp10_hz']) if r['lp10_hz'] else 'n/a'}")
    print("  octave-ish band gains (dB, rel max):")
    for k, v in r["bands"].items():
        print(f"      {k:>8s} Hz : {v:+6.1f}")

    print()
    print("=" * 78)
    print("REAL SESSIONS + raw recording")
    print("=" * 78)
    print(f"{'file':32s} {'centroid':>9s} {'rolloff':>8s} {'flat':>6s} {'RMSdB':>7s} {'dynRng':>7s} {'~SNR':>6s}")
    sessions = sorted(LK.glob("ww_session*.wav")) + [LK / "spectral_search" / "recording.wav"]
    for f in sessions:
        if not f.exists():
            continue
        x = load(f)
        s = spectral_stats(x)
        p = reverb_proxy(x)
        print(f"{f.name:32s} {s['centroid']:8.0f}H {s['rolloff']:7.0f}H "
              f"{s['flatness']:6.3f} {s['rms_db']:6.1f} {p['dyn_range_db']:6.1f} {p['est_snr_db']:5.1f}")

    # plots (optional)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(r["t"] * 1000, r["edc_db"]); ax[0].set_title("Device RIR — Energy Decay (Schroeder)")
        ax[0].set_xlabel("ms"); ax[0].set_ylabel("dB"); ax[0].axhline(-60, ls="--", c="r"); ax[0].grid(True)
        ax[1].semilogx(r["f"], r["Hdb"]); ax[1].set_title("Device RIR — Frequency Response")
        ax[1].set_xlabel("Hz"); ax[1].set_ylabel("dB"); ax[1].set_xlim(50, 8000); ax[1].grid(True)
        fig.tight_layout(); fig.savefig(OUT / "device_rir.png", dpi=110)
        print(f"\nPlot saved: {OUT / 'device_rir.png'}")
    except Exception as e:
        print(f"\n(plots skipped: {e!r})")


if __name__ == "__main__":
    main()
