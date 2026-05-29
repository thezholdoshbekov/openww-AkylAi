"""Live «акылай» wake-word detection for the device (RK3566 / Radxa speaker).

Streams 16 kHz mono audio, runs the openWakeWord chain
(melspectrogram -> Google speech embedding -> akylai head), prints the score and
triggers when it crosses the threshold.

--- Setup on the device (one time) ---
    pip install openwakeword onnxruntime numpy sounddevice soundfile
    # fetch the two base models (melspectrogram + embedding):
    python -c "import openwakeword.utils as u; u.download_models([])"

--- Run ---
    # live from the default microphone:
    python device_detect.py --threshold 0.5

    # if the board is headless ALSA (no PortAudio), use arecord:
    python device_detect.py --source arecord --threshold 0.5

    # sanity-check on a recorded file first (no mic needed):
    python device_detect.py --wav /path/to/test.wav

Notes:
  * The device's 6-mic array exposes the processed/beamformed signal on channel 0
    (that's what the model was tuned on). With --source sd you can pick the input
    device with --device N (list them with: python -c "import sounddevice;print(sounddevice.query_devices())").
  * Start around --threshold 0.5; raise toward 0.9 if you get false triggers.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "akylai_real.onnx"
SR = 16000
CHUNK = 1280  # 80 ms @ 16 kHz — one openWakeWord frame


def gen_sounddevice(device):
    import sounddevice as sd
    with sd.InputStream(samplerate=SR, channels=1, dtype="int16",
                        blocksize=CHUNK, device=device) as stream:
        while True:
            data, _ = stream.read(CHUNK)
            yield data[:, 0].copy()


def gen_arecord():
    cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(SR), "-c", "1", "-t", "raw"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        while True:
            raw = p.stdout.read(CHUNK * 2)        # int16 = 2 bytes
            if len(raw) < CHUNK * 2:
                break
            yield np.frombuffer(raw, dtype=np.int16)
    finally:
        p.terminate()


def gen_wav(path):
    import soundfile as sf
    import librosa
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x[:, 0]
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    x = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    for i in range(0, len(x) - CHUNK, CHUNK):
        yield x[i:i + CHUNK]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--source", choices=["sd", "arecord", "wav"], default="sd")
    ap.add_argument("--wav", default=None, help="wav file when --source wav")
    ap.add_argument("--device", default=None, help="sounddevice input device index (--source sd)")
    args = ap.parse_args()

    from openwakeword.model import Model
    oww = Model(wakeword_models=[args.model], inference_framework="onnx")
    key = list(oww.models.keys())[0]
    print(f"model: {args.model}\nkey: {key}\nthreshold: {args.threshold}\nsource: {args.source}\n"
          "say «акылай» — listening...\n")

    if args.source == "wav":
        if not args.wav:
            sys.exit("--source wav requires --wav PATH")
        stream = gen_wav(args.wav)
    elif args.source == "arecord":
        stream = gen_arecord()
    else:
        dev = int(args.device) if args.device is not None else None
        stream = gen_sounddevice(dev)

    t0 = time.time()
    fired_until = 0.0
    frame = 0
    for chunk in stream:
        score = float(oww.predict(chunk)[key])
        t = frame * CHUNK / SR
        frame += 1
        if score >= args.threshold and t - fired_until > 1.0:
            print(f"  >>> AKYLAI  t={t:6.1f}s  score={score:.3f}")
            fired_until = t
        elif frame % 12 == 0:                       # ~once/sec status
            bar = "#" * int(score * 30)
            print(f"  t={t:6.1f}s  {score:5.3f} |{bar:<30}|", end="\r")


if __name__ == "__main__":
    main()
