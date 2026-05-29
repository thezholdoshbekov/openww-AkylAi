"""Quick evaluation of the trained akylai.onnx.

Tests (peak streaming score per clip):
  1) device_clips (ch0)      — REAL device positives, NOT in training -> recall@thr
  2) bench/benchmark_wavs    — held-out single-utterance clips (akylai + confusables)
  3) bench/user_voice sample — REAL confusable negatives (NOTE: seen in training -> optimistic)
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from openwakeword.model import Model

MODEL_PATH = r"D:/akylai_oww/oww/akylai.onnx"
DEVICE_DIR = Path(__file__).parent / "device_clips"
BENCH_DIR  = Path(r"D:/akylai_oww/data/bench/benchmark_wavs")
UV_DIR     = Path(r"D:/akylai_oww/data/bench/user_voice")
THR = 0.5

oww = Model(wakeword_models=[MODEL_PATH], inference_framework="onnx")
KEY = list(oww.models.keys())[0]
print("model key:", KEY)


def to_int16_16k(path: Path, ch0=False) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = a[:, 0] if ch0 else a.mean(axis=1)
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def peak(path: Path, ch0=False) -> float:
    try:
        data = to_int16_16k(path, ch0=ch0)
    except Exception:                      # ogg etc. -> librosa
        y, _ = librosa.load(str(path), sr=16000, mono=True)
        data = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    oww.reset()
    preds = oww.predict_clip(data, padding=1)
    return max(p[KEY] for p in preds)


# 1) device clips (ch0) — real positives, held-out
print("\n=== device_clips (ch0) — REAL positives (held-out) ===")
dev = sorted(DEVICE_DIR.glob("clip_*.wav"))
dev_peaks = [(p.name, peak(p, ch0=True)) for p in dev]
for n, s in dev_peaks:
    print(f"  {n:14s} peak={s:.3f} {'FIRE' if s >= THR else '.'}")
fired = sum(1 for _, s in dev_peaks if s >= THR)
print(f"  recall@{THR}: {fired}/{len(dev_peaks)} = {fired/len(dev_peaks)*100:.0f}%")

# 2) benchmark wavs — held-out mixed
if BENCH_DIR.exists():
    print("\n=== benchmark_wavs (held-out: akylai + confusables) ===")
    for p in sorted(BENCH_DIR.iterdir()):
        if p.suffix.lower() in (".wav", ".ogg"):
            print(f"  {p.name:24s} peak={peak(p):.3f}")

# 3) user_voice — known top false-positive words (NOTE: seen in training)
if UV_DIR.exists():
    print("\n=== user_voice — top FP words (NOTE: in training, optimistic) ===")
    watch = ["лайк", "кулай", "ылай", "Мамай", "Алтынай", "чай", "Айбек"]
    for w in watch:
        ps = sorted(UV_DIR.glob(f"{w}_*.wav"))
        if ps:
            sc = [peak(p) for p in ps]
            print(f"  {w:10s} peaks={[f'{s:.2f}' for s in sc]}  max={max(sc):.3f}")
