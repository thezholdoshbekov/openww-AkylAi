"""Full device pipeline: mic + TTS-reference  →  DTLN-aec  →  «акылай» wake-word.

This is the runner that should sit between the mic capture and the smart-speaker
agent on the colонka (RK3566). It feeds 8-ms blocks through the neural AEC,
then the cleaned 80-ms frames go to the wake-word model.

Offline mode (validation, no mic needed):
    python aec/device_pipeline.py --mic-wav host_mic.wav --ref-wav host_ref.wav

For live capture on the device the entry-point is the same — replace the file
generators with your mic/TTS-tap streams (sounddevice multichannel or arecord +
TTS-pre-DAC), preserving 16 kHz mono and 128-sample (8 ms) blocks.

Models needed:
  aec/models/dtln_aec_512_1.onnx           DTLN-aec part 1
  aec/models/dtln_aec_512_2.onnx           DTLN-aec part 2
  training/akylai/models/akylai_real.onnx  our wake-word head
  (melspectrogram + embedding base models are downloaded by openwakeword on first run)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from dtln_aec import DTLNAec, BLOCK_SHIFT       # 128-sample (8 ms) hop

SR = 16000
WW_CHUNK = 1280                                  # 80 ms — one openWakeWord frame = 10 AEC blocks


def gen_files(mic_path: str, ref_path: str):
    import soundfile as sf
    mic, sm = sf.read(mic_path, dtype="int16")
    ref, sr_r = sf.read(ref_path, dtype="int16")
    assert sm == SR and sr_r == SR, f"need {SR} Hz wavs (mic={sm} ref={sr_r})"
    n = min(len(mic), len(ref)) // BLOCK_SHIFT
    mic_f = mic.astype(np.float32) / 32768.0
    ref_f = ref.astype(np.float32) / 32768.0
    for i in range(n):
        s = slice(i * BLOCK_SHIFT, (i + 1) * BLOCK_SHIFT)
        yield mic_f[s], ref_f[s]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic-wav", required=True, help="near-end mic wav (16 kHz mono int16)")
    ap.add_argument("--ref-wav", required=True, help="far-end TTS reference wav (16 kHz mono int16)")
    ap.add_argument("--m1", default=str(HERE / "models" / "dtln_aec_512_1.onnx"))
    ap.add_argument("--m2", default=str(HERE / "models" / "dtln_aec_512_2.onnx"))
    ap.add_argument("--wake-model", default=str(REPO / "training" / "akylai" / "models" / "akylai_real.onnx"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--save-clean", default=None, help="optional path to write the AEC output")
    args = ap.parse_args()

    aec = DTLNAec(args.m1, args.m2)
    from openwakeword.model import Model
    oww = Model(wakeword_models=[args.wake_model], inference_framework="onnx")
    ww_key = list(oww.models.keys())[0]
    print(f"AEC: DTLN-aec (block_shift={BLOCK_SHIFT})  WW: {Path(args.wake_model).name}  thr={args.threshold}\n")

    clean_buf = np.zeros(WW_CHUNK, dtype=np.int16)
    fill = 0
    cleaned_all = []
    fired_until = -10.0
    t_block = 0

    for mic_b, ref_b in gen_files(args.mic_wav, args.ref_wav):
        clean = aec.process(mic_b, ref_b)                          # 128 float32 samples
        clean_i16 = (np.clip(clean, -1, 1) * 32767).astype(np.int16)
        cleaned_all.append(clean_i16)

        # Pack 10 AEC blocks (1280 samp) into one wake-word frame
        take = min(WW_CHUNK - fill, len(clean_i16))
        clean_buf[fill:fill + take] = clean_i16[:take]
        fill += take
        if fill == WW_CHUNK:
            score = float(oww.predict(clean_buf)[ww_key])
            t = (t_block * BLOCK_SHIFT) / SR
            t_block += WW_CHUNK // BLOCK_SHIFT                     # advance 10 AEC blocks
            if score >= args.threshold and t - fired_until > 1.0:
                print(f"  >>> AKYLAI  t={t:6.1f}s  score={score:.3f}")
                fired_until = t
            else:
                bar = "#" * int(score * 30)
                print(f"  t={t:6.1f}s  {score:5.3f} |{bar:<30}|", end="\r")
            fill = 0

    print()
    if args.save_clean:
        import soundfile as sf
        sf.write(args.save_clean, np.concatenate(cleaned_all), SR, subtype="PCM_16")
        print(f"AEC output saved -> {args.save_clean}")


if __name__ == "__main__":
    main()
