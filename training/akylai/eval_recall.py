"""Streaming recall test on long device recordings (ww_recall_2026/2027.wav),
mirroring the livekit ww_score methodology so we can compare to that md report.

For each model x file: peak score, #steps>=0.25, #steps>=0.5, and distinct strong
detection events (clustered) with timestamps. openWakeWord streams 1280-sample (80 ms)
chunks; score timeline analyzed the same way as the livekit 2s/100ms sweep.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, soundfile as sf, librosa

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from openwakeword.model import Model

OWW = Path(r"D:/akylai_oww/oww")
MODELS = {"real": OWW / "akylai_real.onnx", "cons": OWW / "akylai_cons.onnx", "w10": OWW / "akylai_w10.onnx"}
FILES = [REPO / "examples/audio/ww_recall_2026.wav", REPO / "examples/audio/ww_recall_2027.wav"]
STEP = 1280  # 80 ms
T_LOW, T_HIGH = 0.25, 0.5


def load_i16(p):
    a, sr = sf.read(str(p), dtype="float32", always_2d=True); x = a[:, 0]
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def events(scores, dt, thr=T_HIGH, gap=0.5):
    """Cluster consecutive windows >= thr into events (merge gaps < `gap` s)."""
    ev, cur = [], None
    last_t = -10
    for i, s in enumerate(scores):
        t = i * dt
        if s >= thr:
            if cur and t - last_t <= gap:
                cur["end"] = t; cur["peak"] = max(cur["peak"], s)
            else:
                if cur:
                    ev.append(cur)
                cur = {"start": t, "end": t, "peak": s}
            last_t = t
    if cur:
        ev.append(cur)
    return ev


for tag, mp in MODELS.items():
    oww = Model(wakeword_models=[str(mp)], inference_framework="onnx")
    KEY = list(oww.models.keys())[0]
    print(f"\n{'='*70}\n MODEL {tag}  ({mp.name})\n{'='*70}")
    for f in FILES:
        data = load_i16(f)
        oww.reset()
        preds = oww.predict_clip(data, padding=0, chunk_size=STEP)
        scores = np.array([d[KEY] for d in preds])
        dt = STEP / 16000
        ev = events(scores, dt, T_HIGH)
        print(f"\n{f.name}  ({len(data)/16000:.0f}s)")
        print(f"  peak={scores.max():.3f}   steps>=0.25={int((scores>=T_LOW).sum())}"
              f"   steps>=0.5={int((scores>=T_HIGH).sum())}   strong events={len(ev)}")
        for k, e in enumerate(ev, 1):
            print(f"    #{k}: t={e['start']:.1f}-{e['end']:.1f}s  peak={e['peak']:.3f}")
