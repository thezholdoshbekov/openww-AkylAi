"""Honest held-out eval on REAL «акылай» voices (Simonlob/Wake_word_Akylai test split):
  positive_real_test (58 live positives)  +  negative_real_test (101 live negatives).

For every *.onnx in D:/akylai_oww/oww: per-clip peak score, then recall (positives that
fire) and false-alarm rate (negatives that fire) over a few thresholds. This tells us, on
REAL voices, whether adding the live positives improved the real recall vs false-alarms.

Run:
    python training/akylai/eval_real.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, soundfile as sf, librosa

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from openwakeword.model import Model

OWW = Path(r"D:/akylai_oww/oww")
POS = OWW / "akylai" / "positive_real_test"
NEG = OWW / "akylai" / "negative_real_test"
THRS = [0.3, 0.5, 0.7, 0.9]


def peaks(model_path, folder):
    m = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
    k = list(m.models.keys())[0]
    out = []
    for p in sorted(folder.glob("real_*.wav")):
        x, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != 16000:
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        data = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        m.reset()
        out.append(max(d[k] for d in m.predict_clip(data, padding=1)))
    return np.array(out)


print(f"real positives={len(list(POS.glob('real_*.wav')))}  real negatives={len(list(NEG.glob('real_*.wav')))}\n")
print(f"{'model':18s}{'thr':>6}{'recall':>9}{'FA-rate':>9}")
print("-" * 42)
for mp in sorted(OWW.glob("*.onnx")):
    ps = peaks(mp, POS)
    ns = peaks(mp, NEG)
    for t in THRS:
        rec = (ps >= t).mean() * 100
        fa = (ns >= t).mean() * 100
        print(f"{mp.stem:18s}{t:6.1f}{rec:8.1f}%{fa:8.1f}%")
    print("-" * 42)
