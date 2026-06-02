"""Offline ERLE comparison: DTLN-aec (neural ONNX) vs WebRTC APM (classic DSP),
on the same recorded mic+ref pair. The de-risk gate from AEC3_PLAN.md, extended
to the neural option.

Inputs (same as aec_offline_test.py):
  host_mic.wav  — raw ch1 (XMOS processing OFF), 16 kHz mono int16
  host_ref.wav  — reference (TTS or ch5 loopback), 16 kHz mono int16
  both sample-synchronous (recorded together with no near-end speech).

Usage:
  python aec_neural_test.py host_mic.wav host_ref.wav --delay-ms 35

ERLE = 10*log10(var(mic) / var(out)). Higher = more echo cancelled.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from dtln_aec import DTLNAec, BLOCK_SHIFT
from aec_offline_test import align, erle_db, run_webrtc, write_wav  # reuse


SR = 16000
HERE = Path(__file__).resolve().parent
M1 = HERE / "models" / "dtln_aec_512_1.onnx"
M2 = HERE / "models" / "dtln_aec_512_2.onnx"


def run_dtln(mic: np.ndarray, ref: np.ndarray) -> np.ndarray | None:
    if not (M1.exists() and M2.exists()):
        print(f"[dtln] models not found at {M1} / {M2} — "
              "download from https://github.com/breizhn/DTLN-aec (pretrained_models/) "
              "and put them under aec/models/")
        return None
    aec = DTLNAec(str(M1), str(M2))
    mic_f = mic.astype(np.float32) / 32768.0
    ref_f = ref.astype(np.float32) / 32768.0
    n = min(len(mic_f), len(ref_f)) // BLOCK_SHIFT
    out = np.zeros(n * BLOCK_SHIFT, dtype=np.float32)
    for i in range(n):
        s = slice(i * BLOCK_SHIFT, (i + 1) * BLOCK_SHIFT)
        out[s] = aec.process(mic_f[s], ref_f[s])
    return (np.clip(out, -1, 1) * 32767).astype(np.int16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mic"); ap.add_argument("ref")
    ap.add_argument("--delay-ms", type=int, default=35,
                    help="mic-after-ref delay (host_aec_diag.py found 35 ms)")
    args = ap.parse_args()

    mic, sr_m = sf.read(args.mic, dtype="int16")
    ref, sr_r = sf.read(args.ref, dtype="int16")
    assert sr_m == SR and sr_r == SR, f"need {SR} Hz (got mic={sr_m} ref={sr_r})"
    delay_samp = int(round(args.delay_ms / 1000 * SR))
    mic_a, ref_a = align(mic, ref, delay_samp)
    print(f"mic={len(mic)} ref={len(ref)}  aligned len={len(mic_a)}  delay={args.delay_ms} ms\n")

    rows = []
    for name, fn in (("WebRTC APM", lambda: run_webrtc(mic_a, ref_a, args.delay_ms)),
                     ("DTLN-aec",   lambda: run_dtln(mic_a, ref_a))):
        out = fn()
        if out is None:
            continue
        e = erle_db(mic_a, out)
        tag = name.split()[0].lower().replace("-", "")
        path = f"clean_{tag}.wav"
        write_wav(path, out)
        rows.append((name, e, path))

    if not rows:
        sys.exit("Ни WebRTC, ни DTLN не отработали. См. aec/AEC3_PLAN.md (Setup) и README DTLN-aec.")

    print(f"\n{'method':14s}{'ERLE (dB)':>12}   output")
    print("-" * 48)
    for n, e, p in rows:
        print(f"{n:14s}{e:+12.1f}   {p}")
    print("\nЦель: ERLE >= 6 dB. Линейный потолок был ~+0.5 dB.")


if __name__ == "__main__":
    main()
