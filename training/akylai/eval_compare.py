"""Compare two trained models (w10 backup vs current w150) on:
  - device_clips (ch0) recall
  - in-distribution test POS/NEG (feature-level)
  - confusable peaks (user_voice + benchmark confusables)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, soundfile as sf, librosa, onnxruntime as ort

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from openwakeword.model import Model

OWW = Path(r"D:/akylai_oww/oww")
MODELS = {"w10": str(OWW / "akylai_w10.onnx"), "w150": str(OWW / "akylai.onnx")}
DEVICE_DIR = Path(__file__).parent / "device_clips"
UV = Path(r"D:/akylai_oww/data/bench/user_voice")
BENCH = Path(r"D:/akylai_oww/data/bench/benchmark_wavs")
THR = 0.5

pos_feat = np.load(str(OWW / "akylai" / "positive_features_test.npy"))
neg_feat = np.load(str(OWW / "akylai" / "negative_features_test.npy"))


def to_i16(path: Path, ch0=False):
    try:
        a, sr = sf.read(str(path), dtype="float32", always_2d=True)
        x = a[:, 0] if ch0 else a.mean(1)
    except Exception:
        x, sr = librosa.load(str(path), sr=16000, mono=True)
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def feat_frac(onnx_path, feats, k=800):
    sess = ort.InferenceSession(onnx_path); iname = sess.get_inputs()[0].name
    idx = np.random.default_rng(0).choice(len(feats), min(k, len(feats)), replace=False)
    s = np.array([float(sess.run(None, {iname: feats[i][None].astype(np.float32)})[0].ravel()[0]) for i in idx])
    return s


CONF = ["лайк", "кулай", "ылай", "алтынай", "асылай", "николай", "Мамай"]

for tag, path in MODELS.items():
    print(f"\n{'='*60}\n MODEL {tag}  ({Path(path).name})\n{'='*60}")
    # feature-level
    ps, ns = feat_frac(path, pos_feat), feat_frac(path, neg_feat)
    print(f"in-dist  POS>0.5={ (ps>THR).mean()*100:5.1f}%   NEG>0.5={ (ns>THR).mean()*100:4.1f}%")

    oww = Model(wakeword_models=[path], inference_framework="onnx")
    KEY = list(oww.models.keys())[0]
    def peak(p, ch0=False):
        oww.reset()
        return max(d[KEY] for d in oww.predict_clip(to_i16(p, ch0=ch0), padding=1))

    dev = sorted(DEVICE_DIR.glob("clip_*.wav"))
    dpk = [peak(p, ch0=True) for p in dev]
    fired = sum(1 for s in dpk if s >= THR)
    print(f"device   recall@{THR}={fired}/{len(dpk)}  (min={min(dpk):.2f} mean={np.mean(dpk):.2f})")

    print("confusables (max peak over reps; >0.5 = FALSE POSITIVE):")
    for w in CONF:
        ps_ = sorted(UV.glob(f"{w}_*.wav"))
        if ps_:
            m = max(peak(p) for p in ps_)
            print(f"   {w:9s} max={m:.2f} {'<-- FP' if m>=THR else 'ok'}")
