"""livekit-style evaluation for our openWakeWord akylai models.

Replicates livekit_ww_akylai/src/livekit/wakeword/eval/evaluate.py metrics, standalone:
  - DET curve (FPR vs FNR over 1001 thresholds) + AUT (area under DET, lower=better)
  - FPPH (false positives per hour) measured on the ~11.3 h general validation set
  - recall @ fixed 0.5 and @ optimal threshold (max recall s.t. FPPH <= target)
  - BONUS: real-world device_clips recall (held-out, ch0)

Runs for every *.onnx in D:/akylai_oww/oww (akylai_w10, akylai_w150, akylai=current),
prints a comparison table, and writes a combined DET plot.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

OWW = Path(r"D:/akylai_oww/oww")
FEAT = OWW / "akylai"
VAL = Path(r"D:/akylai_oww/validation_set_features.npy")
DEVICE_DIR = Path(__file__).parent / "device_clips"
OUT = Path(r"D:/akylai_oww/eval"); OUT.mkdir(exist_ok=True)
VAL_HOURS = 11.3
TARGET_FPPH = 1.0

pos = np.load(str(FEAT / "positive_features_test.npy"))
neg_adv = np.load(str(FEAT / "negative_features_test.npy"))
val = np.load(str(VAL))
if val.ndim == 2:
    val = val[: (val.shape[0] // 16) * 16].reshape(-1, 16, 96)
print(f"pos={pos.shape[0]}  neg_adv={neg_adv.shape[0]}  val_general={val.shape[0]} (~{VAL_HOURS}h)")


def predict(sess, feats):
    iname = sess.get_inputs()[0].name
    f32 = feats.astype(np.float32)
    out = np.empty(len(f32), dtype=np.float32)
    for i in range(len(f32)):
        out[i] = sess.run(None, {iname: f32[i:i+1]})[0].ravel()[0]
    return out


def device_recall(model_path, thr=0.5):
    """Real held-out device positives (ch0) recall."""
    import soundfile as sf, librosa
    from openwakeword.model import Model
    m = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
    k = list(m.models.keys())[0]
    fired = 0; clips = sorted(DEVICE_DIR.glob("clip_*.wav"))
    for p in clips:
        a, sr = sf.read(str(p), dtype="float32", always_2d=True); x = a[:, 0]
        if sr != 16000:
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        data = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        m.reset()
        pk = max(d[k] for d in m.predict_clip(data, padding=1))
        fired += pk >= thr
    return fired, len(clips)


rows = []
plt.figure(figsize=(8, 7))
for mp in sorted(OWW.glob("*.onnx")):
    sess = ort.InferenceSession(str(mp), providers=["CPUExecutionProvider"])
    ps = predict(sess, pos)
    ns_gen = predict(sess, val)
    ns = np.concatenate([predict(sess, neg_adv), ns_gen])

    thr = np.linspace(0, 1, 1001)
    fpr = np.array([(ns >= t).mean() for t in thr])
    fnr = np.array([(ps < t).mean() for t in thr])
    si = np.argsort(fpr)
    aut = float(np.trapz(fnr[si], fpr[si]))
    fpph = np.array([(ns_gen >= t).sum() / VAL_HOURS for t in thr])

    # fixed 0.5
    r05 = (ps >= 0.5).mean(); f05 = (ns_gen >= 0.5).sum() / VAL_HOURS
    # optimal: max recall s.t. fpph <= target
    ok = np.where(fpph <= TARGET_FPPH)[0]
    bi = ok[np.argmax([(ps >= thr[i]).mean() for i in ok])] if len(ok) else np.argmin(fpph)
    r_opt = (ps >= thr[bi]).mean(); f_opt = fpph[bi]

    dev_f, dev_n = device_recall(mp)
    rows.append((mp.stem, aut, r05 * 100, f05, thr[bi], r_opt * 100, f_opt, dev_f, dev_n))
    plt.plot(fpr[si] * 100, fnr[si] * 100, lw=2, label=f"{mp.stem} (AUT={aut:.3f})")

plt.plot([0, 100], [100, 0], "--", color="gray", alpha=0.5)
plt.xlabel("False Positive Rate (%)"); plt.ylabel("False Negative Rate (%)")
plt.title("DET — akylai models"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(str(OUT / "det_compare.png"), dpi=140); plt.close()

print("\n" + "=" * 92)
print(f"{'model':16s}{'AUT':>8}{'rec@.5':>9}{'fpph@.5':>9}{'optThr':>8}{'rec@opt':>9}{'fpph@opt':>10}{'device':>9}")
print("-" * 92)
for n, aut, r05, f05, ot, ro, fo, df, dn in rows:
    print(f"{n:16s}{aut:8.3f}{r05:8.1f}%{f05:9.2f}{ot:8.2f}{ro:8.1f}%{fo:10.2f}{df:>6}/{dn}")
print("=" * 92)
print(f"DET plot -> {OUT/'det_compare.png'}   (AUT lower=better; fpph=false alarms/hour on {VAL_HOURS}h)")
