"""Definitive clock test: measure the exact frequency of a played 1kHz tone on
ch1 (acoustic echo, driven by whatever DAC feeds the speaker) vs ch5 (the XMOS
reference). Both are captured by the SAME ReSpeaker ADC, so any frequency
difference between them reflects the difference between the SOURCE clocks:
  - same freq  -> speaker DAC and reference on one clock (AEC viable)
  - diff freq  -> two clocks; |df|/f gives the drift in ppm (AEC impossible)

Files: tone_ch1.wav, tone_ch5.wav (16k mono, a 1000 Hz sine recorded ~20s).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

SR = 16000
F0 = 1000.0


def load(name):
    x, sr = sf.read(name, dtype="float64", always_2d=True)
    x = x[:, 0]
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x


def fine_freq(x):
    """FFT peak with parabolic interpolation for sub-bin frequency."""
    x = x - x.mean()
    w = np.hanning(len(x))
    X = np.abs(np.fft.rfft(x * w))
    f = np.fft.rfftfreq(len(x), 1 / SR)
    k = int(np.argmax(X))
    if 0 < k < len(X) - 1:
        a, b, c = X[k - 1], X[k], X[k + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)  # parabolic offset in bins
    else:
        d = 0.0
    df = f[1] - f[0]
    return (k + d) * df


def main() -> None:
    p1, p5 = Path("tone_ch1.wav"), Path("tone_ch5.wav")
    if not (p1.exists() and p5.exists()):
        print("нет tone_ch1.wav / tone_ch5.wav — принеси файлы тонового теста.")
        return
    x1, x5 = load(str(p1)), load(str(p5))
    n = min(len(x1), len(x5))
    x1, x5 = x1[:n], x5[:n]
    f1, f5 = fine_freq(x1), fine_freq(x5)
    print(f"dur={n/SR:.1f}s   FFT bin={SR/n:.4f} Hz")
    print(f"ch1 (acoustic echo) tone = {f1:.4f} Hz")
    print(f"ch5 (XMOS reference) tone = {f5:.4f} Hz")
    diff = f1 - f5
    ppm = diff / F0 * 1e6
    print(f"\nΔf = {diff:+.4f} Hz   ->  clock offset = {ppm:+.1f} ppm")
    print("\n=== ВЕРДИКТ ===")
    if abs(ppm) < 10:
        print(f"|{ppm:+.1f}| ppm < 10 -> ОДИН КЛОК. Разводка правильная, AEC может работать.")
        print("   (остаточная разница в пределах точности измерения)")
    else:
        drift_s = abs(diff) / F0  # fraction; samples drift per second at 16k
        print(f"{ppm:+.1f} ppm -> ДВА КЛОКА подтверждены численно.")
        print(f"   дрейф ~{drift_s*SR:.2f} сэмпла/сек -> ~{drift_s*SR*60:.0f} сэмплов/мин.")
        print("   HW AEC не сойдётся: путь эха уезжает быстрее, чем фильтр адаптируется.")


if __name__ == "__main__":
    main()
