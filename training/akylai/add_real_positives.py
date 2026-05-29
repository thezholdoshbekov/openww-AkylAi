"""Inject the REAL-voice «акылай» recordings (Simonlob/Wake_word_Akylai) into the
openWakeWord training folders, alongside the 24k TTS positives.

Source : D:/akylai_oww/data_hf/data/{train,test}-00000-of-00001.parquet
         train: 667 positive (label=1) + 761 negative (label=0)
         test :  58 positive       +  101 negative

What it does:
  * train positives -> oww/akylai/positive_train/real_*.wav, DUPLICATED x DUP
      (duplication = upweight: each copy gets different random augmentation in the
       feature step, so 667 real voices get ~DUPx the influence vs the 24k TTS pool)
  * train negatives -> oww/akylai/negative_train/real_*.wav  (x1)
  * test  pos/neg   -> oww/akylai/positive_real_test / negative_real_test  (x1, HONEST
       held-out real eval — kept OUT of the TTS positive_test so eval_det stays comparable)

Re-runnable: clears previous real_*.wav first, never touches the TTS clip_*.wav.
augment_features.py globs *.wav, so it will pick up real_*.wav automatically.

Run:
    python training/akylai/add_real_positives.py --dup 10
then recompute TRAIN features and retrain:
    python training/akylai/augment_features.py --train-only
    python openwakeword/train.py --training_config training/akylai/akylai.yml --train_model
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

WORK = Path(r"D:/akylai_oww")
HF = WORK / "data_hf" / "data"
OWW = WORK / "oww" / "akylai"
SR = 16000


def decode(b: bytes) -> np.ndarray:
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x


def wav_bytes(x: np.ndarray) -> bytes:
    out = io.BytesIO()
    sf.write(out, x, SR, format="WAV", subtype="PCM_16")
    return out.getvalue()


def clear_real(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for f in folder.glob("real_*.wav"):
        f.unlink()


def rows(split: str):
    t = pq.read_table(HF / f"{split}-00000-of-00001.parquet")
    au = t.column("audio")
    lab = t.column("label").to_pylist()
    return au, lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dup", type=int, default=10,
                    help="how many augmented copies of each real positive to add (upweight)")
    args = ap.parse_args()

    pos_train = OWW / "positive_train"
    neg_train = OWW / "negative_train"
    pos_rtest = OWW / "positive_real_test"
    neg_rtest = OWW / "negative_real_test"
    for d in (pos_rtest, neg_rtest):
        clear_real(d)
    clear_real(pos_train)
    clear_real(neg_train)

    # ---- TRAIN ----
    au, lab = rows("train")
    n_pos = n_neg = 0
    for i, l in enumerate(lab):
        x = decode(au[i].as_py()["bytes"])
        if l == 1:
            for c in range(args.dup):
                (pos_train / f"real_{n_pos:06d}.wav").write_bytes(wav_bytes(x))
                n_pos += 1
        else:
            (neg_train / f"real_{n_neg:06d}.wav").write_bytes(wav_bytes(x))
            n_neg += 1
    print(f"train: +{n_pos} positive wavs (667 x{args.dup})   +{n_neg} negative wavs")

    # ---- TEST (honest real held-out) ----
    au, lab = rows("test")
    tp = tn = 0
    for i, l in enumerate(lab):
        x = decode(au[i].as_py()["bytes"])
        if l == 1:
            (pos_rtest / f"real_{tp:06d}.wav").write_bytes(wav_bytes(x)); tp += 1
        else:
            (neg_rtest / f"real_{tn:06d}.wav").write_bytes(wav_bytes(x)); tn += 1
    print(f"real test: {tp} positive -> {pos_rtest.name}   {tn} negative -> {neg_rtest.name}")
    print("\nNext:\n  python training/akylai/augment_features.py --train-only"
          "\n  python openwakeword/train.py --training_config training/akylai/akylai.yml --train_model")


if __name__ == "__main__":
    main()
