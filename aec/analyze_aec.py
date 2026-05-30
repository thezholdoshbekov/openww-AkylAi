"""Verdict on whether HW AEC works after rewiring the speaker to the ReSpeaker
3.5mm output (single XMOS clock). Reads aectest_ch0/ch1/ch5.wav (16k mono each,
split from a 6ch capture taken WHILE the agent plays speech and the user is silent).

Two metrics:
  1) coherence ch1(echo) <-> ch5(reference): was 0.18 (two clocks, AEC impossible).
     If now >= 0.5 -> the clock-domain problem is fixed, AEC CAN converge.
  2) ERLE = RMS(ch1) - RMS(ch0) over windows where the reference (ch5) is active:
     was ~ -5 dB (ch0 LOUDER than raw). If now strongly positive (+10..20 dB) ->
     echo is actually being cancelled on ch0.
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


def load(c):
    f = Path(f"aectest_ch{c}.wav")
    x, sr = sf.read(str(f), dtype="float32", always_2d=True)
    x = x[:, 0]
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x


def db(v):
    return 20 * np.log10(max(v, 1e-9))


def main() -> None:
    ch0, ch1, ch5 = load(0), load(1), load(5)
    n = min(len(ch0), len(ch1), len(ch5))
    ch0, ch1, ch5 = ch0[:n], ch1[:n], ch5[:n]
    print(f"dur={n/SR:.1f}s")

    # ---- 1) global cross-correlation peak (normalized) ----
    a = (ch1 - ch1.mean()) / (ch1.std() + 1e-9)
    b = (ch5 - ch5.mean()) / (ch5.std() + 1e-9)
    maxlag = int(0.06 * SR)
    xc = np.correlate(a, b, mode="full") / n
    mid = len(xc) // 2
    seg = xc[mid - maxlag : mid + maxlag + 1]
    lags = np.arange(-maxlag, maxlag + 1) / SR * 1000
    k = int(np.argmax(np.abs(seg)))
    coh = abs(seg[k])
    print(f"\n[1] coherence ch1<->ch5 = {coh:.3f}  @ lag {lags[k]:+.1f} ms"
          f"   (было 0.18; нужно >=0.5)")
    erle_ceiling = -10 * np.log10(max(1 - coh ** 2, 1e-9))
    print(f"    теоретический потолок ERLE при этой coherence = {erle_ceiling:.1f} dB")

    # ---- 2) ERLE over windows where the reference is active ----
    win = int(0.25 * SR)
    rows = []
    ref_levels = []
    for i in range(n // win):
        s = slice(i * win, (i + 1) * win)
        r0 = np.sqrt(np.mean(ch0[s] ** 2))
        r1 = np.sqrt(np.mean(ch1[s] ** 2))
        r5 = np.sqrt(np.mean(ch5[s] ** 2))
        rows.append((i * 0.25, db(r0), db(r1), db(r5)))
        ref_levels.append(db(r5))
    ref_levels = np.array(ref_levels)
    # "agent speaking" = reference louder than its 60th percentile
    thr = np.percentile(ref_levels, 60)
    active = [(t, e0, e1, e5) for (t, e0, e1, e5) in rows if e5 >= thr]
    if active:
        erle = np.mean([e1 - e0 for (_, e0, e1, _) in active])
        print(f"\n[2] ERLE (ch1-ch0) во время речи агента = {erle:+.1f} dB"
              f"   (было -5; нужно сильно +)")
    print("\nt(s)   ch0(AEC)  ch1(raw)  ch5(ref)   ch1-ch0")
    for (t, e0, e1, e5) in rows:
        mark = " <-agent" if e5 >= thr else ""
        print(f"{t:4.1f}   {e0:7.1f}  {e1:7.1f}  {e5:7.1f}   {e1-e0:+6.1f}{mark}")

    print("\n=== ВЕРДИКТ ===")
    if coh >= 0.5:
        print(f"coherence {coh:.2f} >= 0.5 -> КЛОК ОДИН, разводка правильная, AEC может работать.")
    elif coh >= 0.3:
        print(f"coherence {coh:.2f} -> частично связались; лучше, но не идеал. Проверить роутинг playback.")
    else:
        print(f"coherence {coh:.2f} -> всё ещё декоррелировано: звук, похоже, играл НЕ через ReSpeaker.")


if __name__ == "__main__":
    main()
