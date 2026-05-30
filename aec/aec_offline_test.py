"""De-risk gate for the custom AEC: run a NONLINEAR echo canceller (WebRTC APM
first, SpeexDSP fallback) on an already-recorded synchronous session and measure
ERLE. This tells us whether a real AEC beats the linear-only ceiling (~0.5 dB at
corr 0.31) BEFORE we invest in real-time integration on the Radxa.

Inputs (from Arsen's host_aec_record.sh): a raw-mic wav and a reference wav,
both 16 kHz mono, sample-synchronous (same arecord session / same clock).
  host_mic.wav  — raw ch1 (XMOS processing OFF)
  host_ref.wav  — ch5 USB loopback   (or the host TTS stream resampled to 16k)

Usage:
  python aec_offline_test.py host_mic.wav host_ref.wav --delay-ms 35

Reads the mic↔ref delay from host_aec_diag.py (they found +561 samples = 35 ms).
ERLE is measured over the whole file (the record session has NO near-end speech,
so all mic energy is echo -> ERLE = 10log10(var(mic)/var(out))).

Run on the device/Lenovo where webrtc or speex is installed (see AEC3_PLAN.md).
"""
from __future__ import annotations

import argparse
import sys
import wave

import numpy as np

SR = 16000
FRAME = 160  # 10 ms @ 16 kHz — both WebRTC APM and Speex want this


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR, f"{path}: need 16 kHz, got {w.getframerate()}"
        assert w.getnchannels() == 1, f"{path}: need mono"
        n = w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype=np.int16)
    return x


def write_wav(path: str, x: np.ndarray) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.clip(x, -32768, 32767).astype(np.int16).tobytes())


def erle_db(mic: np.ndarray, out: np.ndarray) -> float:
    n = min(len(mic), len(out))
    vm = np.var(mic[:n].astype(np.float64))
    vo = np.var(out[:n].astype(np.float64))
    return 10 * np.log10(max(vm, 1e-9) / max(vo, 1e-9))


def align(mic: np.ndarray, ref: np.ndarray, delay_samples: int):
    """mic is delay_samples LATER than ref. Drop the leading ref so frame i of
    ref lines up with the echo it produced in mic."""
    if delay_samples > 0:
        ref = ref[delay_samples:]
    elif delay_samples < 0:
        mic = mic[-delay_samples:]
    n = min(len(mic), len(ref))
    return mic[:n].copy(), ref[:n].copy()


def estimate_drift_ppm(mic: np.ndarray, ref: np.ndarray,
                       win_s=3.0, hop_s=1.5, maxlag_ms=60) -> tuple[float, float]:
    """Two-clock drift estimate. Best-lag(mic,ref) per window should drift
    LINEARLY with time if the clocks differ. Slope (samples/s) -> ppm.
    Returns (ppm, fraction_of_windows_used). Needs a long-ish session (>=20 s)
    and enough echo energy; on a single-clock session ppm ~ 0."""
    from scipy.signal import correlate
    a = mic.astype(np.float64); b = ref.astype(np.float64)
    win = int(win_s * SR); hop = int(hop_s * SR); maxlag = int(maxlag_ms / 1000 * SR)
    ts, lags = [], []
    attempted = 0
    for s in range(0, min(len(a), len(b)) - win, hop):
        attempted += 1
        x = a[s:s + win]; y = b[s:s + win]
        x = (x - x.mean()) / (x.std() + 1e-9); y = (y - y.mean()) / (y.std() + 1e-9)
        xc = correlate(x, y, mode="full", method="fft") / win
        mid = len(xc) // 2; seg = xc[mid - maxlag:mid + maxlag + 1]
        k = int(np.argmax(np.abs(seg)))
        if abs(seg[k]) < 0.12:      # skip windows with no reliable peak
            continue
        ts.append((s + win / 2) / SR); lags.append(k - maxlag)
    if len(ts) < 3:
        return 0.0, 0.0
    ts = np.array(ts); lags = np.array(lags)
    # Theil-Sen robust slope (median of pairwise slopes) in samples/sec
    slopes = [(lags[j] - lags[i]) / (ts[j] - ts[i])
              for i in range(len(ts)) for j in range(i + 1, len(ts))]
    slope = float(np.median(slopes))
    ppm = slope / SR * 1e6
    used = len(ts) / max(1, attempted)
    return ppm, used


def resample_ppm(x: np.ndarray, ppm: float) -> np.ndarray:
    """Stretch x by (1 + ppm*1e-6) so it tracks the mic clock (drift comp)."""
    if abs(ppm) < 0.5:
        return x
    n = len(x)
    t = np.arange(n)
    tp = t * (1 + ppm * 1e-6)
    return np.interp(t, tp, x).astype(x.dtype)


def run_webrtc(mic: np.ndarray, ref: np.ndarray, delay_ms: int) -> np.ndarray | None:
    try:
        from webrtc_audio_processing import AudioProcessingModule as AP
    except Exception as e:
        print(f"[webrtc] not available: {e}")
        return None
    ap = AP(enable_ns=True, enable_aec=True, enable_agc=False)
    ap.set_stream_format(SR, 1)
    ap.set_reverse_stream_format(SR, 1)
    try:
        ap.set_ns_level(2)
    except Exception:
        pass
    out = np.zeros_like(mic)
    nfr = min(len(mic), len(ref)) // FRAME
    for i in range(nfr):
        s = slice(i * FRAME, (i + 1) * FRAME)
        ap.process_reverse_stream(ref[s].tobytes())      # far-end (played)
        ap.set_stream_delay_ms(delay_ms)
        y = ap.process_stream(mic[s].tobytes())          # near-end -> cleaned
        out[s] = np.frombuffer(y, dtype=np.int16)
    return out


def run_speex(mic: np.ndarray, ref: np.ndarray) -> np.ndarray | None:
    try:
        from speexdsp import EchoCanceller
    except Exception as e:
        print(f"[speex] not available: {e}")
        return None
    filt = 4096  # ~256 ms — must cover RT60 tail (they measured 0.66 s; longer is better but slower)
    ec = EchoCanceller.create(FRAME, filt, SR)
    out = np.zeros_like(mic)
    nfr = min(len(mic), len(ref)) // FRAME
    for i in range(nfr):
        s = slice(i * FRAME, (i + 1) * FRAME)
        y = ec.process(mic[s].tobytes(), ref[s].tobytes())
        out[s] = np.frombuffer(y, dtype=np.int16)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mic")
    ap.add_argument("ref")
    ap.add_argument("--delay-ms", type=int, default=35,
                    help="mic-after-ref delay (host_aec_diag.py found 35 ms)")
    ap.add_argument("--drift-comp", action="store_true",
                    help="estimate two-clock drift (PCM5102 prod path) and resample ref")
    args = ap.parse_args()

    mic = read_wav(args.mic)
    ref = read_wav(args.ref)
    delay_samples = int(round(args.delay_ms / 1000 * SR))
    mic_a, ref_a = align(mic, ref, delay_samples)
    print(f"mic={len(mic)} ref={len(ref)} samples; aligned len={len(mic_a)} "
          f"(delay {args.delay_ms} ms = {delay_samples} samp)")

    if args.drift_comp:
        ppm, used = estimate_drift_ppm(mic_a, ref_a)
        print(f"drift estimate: {ppm:+.1f} ppm  (from {used*100:.0f}% of windows)")
        if used < 0.3:
            print("  ! too few reliable windows — drift estimate unreliable "
                  "(need longer/louder session). Proceeding without comp.")
        else:
            ref_a = resample_ppm(ref_a, ppm)
            print(f"  applied drift comp ({ppm:+.1f} ppm)")
    print()

    ran = False
    for name, fn in (("WebRTC APM", lambda: run_webrtc(mic_a, ref_a, args.delay_ms)),
                     ("SpeexDSP MDF", lambda: run_speex(mic_a, ref_a))):
        out = fn()
        if out is None:
            continue
        ran = True
        e = erle_db(mic_a, out)
        write_wav(f"clean_{name.split()[0].lower()}.wav", out)
        print(f"=== {name} ===")
        print(f"  ERLE = {e:+.1f} dB   (linear ceiling here was ~+0.5 dB)")
        verdict = ("ПРОБИЛ потолок -> нелинейный AEC работает, идём в real-time"
                   if e >= 6 else
                   "слабо (<6 dB) -> акустику чинить (колонку ближе) или менять сетап")
        print(f"  -> {verdict}\n")
    if not ran:
        print("Ни webrtc, ни speex не установлены — см. aec/AEC3_PLAN.md (раздел Setup).")
        sys.exit(1)


if __name__ == "__main__":
    main()
