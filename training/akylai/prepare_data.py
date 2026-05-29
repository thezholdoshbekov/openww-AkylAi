"""Convert the AkylAi Wake Word V4 parquet dataset into the folder layout that
openWakeWord's train.py expects.  FULL run (not the smoke-test subset).

Source : training/akylai/data  (HF snapshot: positives/, negatives/{confusables-v2,
                                 podcast-cuts,musan}/, backgrounds/podcasts_2s.parquet, manifests/)
Target : training/akylai/oww/akylai/{positive_train,positive_test,negative_train,negative_test}/clip_*.wav
         training/akylai/oww/backgrounds_wav/bg_*.wav      (podcast 2-s cuts, mix-in pool)
         training/akylai/oww/musan_wav/musan_*.wav         (MUSAN noise/music, mix-in pool)

Split + classes come ENTIRELY from the authors' FULL manifests (ww_train / ww_val).
Those manifests reference every shard type and carry the label, so a single
manifest-driven loop writes positives (label=1) and ALL negatives — base negatives,
confusables, podcast-speech, and MUSAN (label=0). No separate per-class loop needed.

Notes:
  * PITCH_SHIFTS is OFF by default at full scale: 24k real positives already cover
    voice variety, and openWakeWord's own augment step applies PitchShift ±3.
  * MUSAN clips appear both as negatives (via manifest) and in the mix-in pool
    (musan_wav) — that double duty matches the dataset design (DESIGN §6.1).
  * Output folders are cleared first so re-runs don't mix old/new clips.

All audio is written mono 16 kHz 16-bit PCM WAV (positives may be 22050 Hz -> resampled;
negatives/backgrounds are already 16 kHz).
"""

from __future__ import annotations

import collections
import io
import wave
from pathlib import Path

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

BASE = Path(__file__).resolve().parent
WORK = Path(r"D:/akylai_oww")        # data + outputs live on D: (C: is nearly full)
DATA = WORK / "data"
OUT = WORK / "oww" / "akylai"
BG_OUT = WORK / "oww" / "backgrounds_wav"
MUSAN_OUT = WORK / "oww" / "musan_wav"
USER_VOICE = DATA / "bench" / "user_voice"   # 365 REAL-voice confusable recordings -> negatives
TARGET_SR = 16000

MANIFESTS = [("train", "ww_train.parquet"), ("val", "ww_val.parquet")]  # FULL split
PITCH_SHIFTS: list[int] = []   # e.g. [-4, 4] to bake extra pitched positive copies; off at full scale
MAX_BACKGROUNDS = 6000         # mix-in pool cap (disk/variety tradeoff)
MAX_MUSAN = 4000               # mix-in pool cap


def decode_16k(wav_bytes: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio


def f32_to_wav_bytes(audio: np.ndarray) -> bytes:
    out = io.BytesIO()
    sf.write(out, audio, TARGET_SR, format="WAV", subtype="PCM_16")
    return out.getvalue()


def build_shard_index() -> dict[str, Path]:
    """Map shard basename -> path, across every dir the manifests can reference."""
    index: dict[str, Path] = {}
    for d in [DATA / "positives",
              DATA / "negatives" / "confusables-v2",
              DATA / "negatives" / "podcast-cuts",
              DATA / "negatives" / "musan"]:
        for p in d.glob("*.parquet"):
            index[p.name] = p
    return index


def dest_folder(split: str, label: int) -> str:
    cls = "positive" if label == 1 else "negative"
    suffix = "train" if split == "train" else "test"
    return f"{cls}_{suffix}"


def write_clip(folder: str, counters: dict[str, int], audio: np.ndarray) -> None:
    n = counters[folder]
    counters[folder] = n + 1
    (OUT / folder / f"clip_{n:06d}.wav").write_bytes(f32_to_wav_bytes(audio))


def _clear(dir_path: Path, pattern: str) -> None:
    if dir_path.exists():
        for f in dir_path.glob(pattern):
            f.unlink()
    dir_path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    for folder in ("positive_train", "negative_train", "positive_test", "negative_test"):
        _clear(OUT / folder, "clip_*.wav")
    _clear(BG_OUT, "bg_*.wav")
    _clear(MUSAN_OUT, "musan_*.wav")

    shard_index = build_shard_index()
    counters: dict[str, int] = collections.defaultdict(int)

    # ---- manifest-driven: positives + ALL negatives, grouped by shard ----
    tasks: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    for split, man_file in MANIFESTS:
        man = pq.read_table(DATA / "manifests" / man_file,
                            columns=["shard_path", "row_index", "label"]).to_pydict()
        for sp, ri, lab in zip(man["shard_path"], man["row_index"], man["label"]):
            tasks[Path(sp).name].append((ri, dest_folder(split, lab)))

    total = sum(len(v) for v in tasks.values())
    written = 0
    missing = 0
    for shard_name, entries in sorted(tasks.items()):
        shard_path = shard_index.get(shard_name)
        if shard_path is None:
            print(f"WARNING: shard {shard_name} not on disk, skipping {len(entries)} rows")
            missing += len(entries)
            continue
        audio_col = pq.read_table(shard_path, columns=["audio"]).column("audio")
        for row_index, folder in entries:
            audio = decode_16k(audio_col[row_index].as_py()["bytes"])
            write_clip(folder, counters, audio)
            written += 1
            if folder == "positive_train" and PITCH_SHIFTS:
                for steps in PITCH_SHIFTS:
                    shifted = librosa.effects.pitch_shift(y=audio, sr=TARGET_SR, n_steps=steps)
                    write_clip(folder, counters, shifted.astype(np.float32))
        print(f"  {shard_name}: {len(entries)} rows  ({written}/{total} base clips)")
    if missing:
        print(f"!! {missing} rows skipped (shards not downloaded — e.g. podcast-cuts)")

    # ---- REAL-voice confusables (bench/user_voice) -> hard negatives, 90/10 ----
    # Real (non-TTS) recordings of 121 confusable words x3 reps; includes the known
    # top false-positives (лайк, кулай, ылай, ох-вай, Мамай...). 16 kHz mono, ~3 s.
    if USER_VOICE.exists():
        rng = np.random.default_rng(0)
        uv = sorted(USER_VOICE.glob("*.wav"))
        for p in uv:
            folder = "negative_test" if rng.random() < 0.1 else "negative_train"
            write_clip(folder, counters, decode_16k(p.read_bytes()))
        print(f"  user_voice (REAL negatives): {len(uv)} clips")
    else:
        print("NOTE: bench/user_voice not found — skipped (real-voice negatives)")

    # ---- mix-in pools (NOT classifier classes): podcast backgrounds + MUSAN ----
    _unpack_pcm_backgrounds(DATA / "backgrounds" / "podcasts_2s.parquet", BG_OUT, "bg", MAX_BACKGROUNDS)

    musan_files = sorted((DATA / "negatives" / "musan").glob("*.parquet"))
    if musan_files:
        n = 0
        for mf in musan_files:
            audio_col = pq.read_table(mf, columns=["audio"]).column("audio")
            for i in range(len(audio_col)):
                if n >= MAX_MUSAN:
                    break
                (MUSAN_OUT / f"musan_{n:06d}.wav").write_bytes(
                    f32_to_wav_bytes(decode_16k(audio_col[i].as_py()["bytes"])))
                n += 1
            if n >= MAX_MUSAN:
                break
        print(f"  musan pool: {n} WAVs")
    else:
        print("NOTE: MUSAN parquet not found — skipped")

    print("\nDone. Clips per folder:")
    for folder in ("positive_train", "negative_train", "positive_test", "negative_test"):
        print(f"  {folder}: {counters[folder]}")


def _unpack_pcm_backgrounds(src: Path, out_dir: Path, prefix: str, cap: int) -> None:
    if not src.exists():
        print(f"WARNING: {src} not found")
        return
    pf = pq.ParquetFile(src)
    n = 0
    for batch in pf.iter_batches(columns=["pcm"], batch_size=1000):
        for pcm in batch.column("pcm").to_pylist():
            if n >= cap:
                break
            buf = io.BytesIO()
            w = wave.open(buf, "wb")
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(TARGET_SR)
            w.writeframes(pcm)
            w.close()
            (out_dir / f"{prefix}_{n:06d}.wav").write_bytes(buf.getvalue())
            n += 1
        if n >= cap:
            break
    print(f"  backgrounds pool: {n} WAVs")


if __name__ == "__main__":
    main()
