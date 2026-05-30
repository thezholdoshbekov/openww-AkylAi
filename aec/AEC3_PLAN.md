# Свой AEC для akylai (barge-in) — план

**Решение:** barge-in обязателен → нелинейный AEC (WebRTC APM/AEC3 или Speex MDF).
Линейный AEC руками — тупик (потолок +0.44 dB при корреляции 0.31, доказано
дважды). Нелинейный residual-подавитель этот потолок пробивает.

**Контекст железа (важно):**
- Production: Radxa + PCM5102 (48 кГц TTS) + ReSpeaker mic → **два клока** (дрейф).
  Значит нужна **компенсация дрейфа** в фронтенде AEC.
- Lab (Lenovo, расследование Арсена): колонка на 3.5мм jack ReSpeaker → 1 клок
  (2.5 ppm). На lab-данных дрейфа нет — годятся для первого де-риска.

---

## Шаг 0 — ДЕ-РИСК (сделать первым, до всякой интеграции)

Прогнать уже записанные `host_mic.wav` + `host_ref.wav` через нелинейный AEC и
померить ERLE. Если пробивает ~6 dB+ — подход рабочий. Если нет — проблема в
акустике (колонка далеко/нелинейна), и никакой софт не спасёт.

```bash
cd ~/wake-word
python aec_offline_test.py host_mic.wav host_ref.wav --delay-ms 35
```
Скрипт: [aec_offline_test.py](aec_offline_test.py). Пробует WebRTC, потом Speex,
пишет `clean_*.wav` и печатает ERLE + вердикт.

**Гейты:**
- ERLE ≥ ~10 dB → отлично, идём в real-time (Шаг 2).
- ERLE 6–10 dB → рабочее, но колонку лучше придвинуть (<20 см) и переснять.
- ERLE < 6 dB → сначала акустика (Шаг 1), софт вторичен.

---

## Шаг 1 — акустика (если де-риск слабый)

Корреляция 0.31 — это акустика, не алгоритм. Поднять её:
- колонка **≤ 20 см** от ReSpeaker;
- колонка с **линейным** усилителем (не консьюмерская с DSP/компрессором —
  он плодит гармоники, которых нет в референсе);
- тише фон. Переснять `host_aec_record.sh` → `host_aec_diag.py` (corr должна
  вырасти к 0.6–0.9). Затем повторить Шаг 0.

---

## Шаг 2 — real-time интеграция на Radxa

Пайплайн:
```
TTS-аудио (то, что играем) ──┬──> PCM5102 48кГц → колонка
                             └──> [resample 16k] → AEC.process_reverse_stream()   (far-end)
ReSpeaker raw ch1 (16k) ───────────────────────> AEC.process_stream()            (near-end)
                                                        │ echo-cancelled
                                                        ├──> STT (чистый юзер)
                                                        └──> wake-word (опц.)
```
- Кадр **10 мс / 160 сэмплов** int16 (требование и WebRTC, и Speex).
- `set_stream_delay_ms(35)` — начальная оценка задержки (host_aec_diag нашёл 35 мс);
  APM дальше доуточняет сам.
- **Far-end = сам TTS-поток** (до PCM5102), ресемпленный в 16 кГц — это чище, чем
  ch5 loopback (у ch5 свой ресемплинг и −27 dB уровень).

### Компенсация дрейфа (production, два клока)
PCM5102 и ReSpeaker-ADC разъезжаются (в lab было 2.5 ppm на одном клоке; на
PCM5102 — мерить тоновым тестом `analyze_tone.py`). Варианты:
1. **Дать APM самому** — WebRTC APM терпит малый дрейф через свой delay-tracker.
   Если измеренный дрейф мал (< ~20 ppm) — может хватить из коробки.
2. **Явный ресемплер** — пересэмплировать far-end с коэффициентом (1 + ppm·1e-6),
   ppm брать из тонового теста и/или подстраивать по дрейфу оценки задержки APM.
   Это та кастомная часть, которой не было в расследовании.

---

## Setup (Radxa, Debian/Ubuntu ARM)

WebRTC APM (предпочтительно — есть нелинейный подавитель):
```bash
sudo apt install -y swig libwebrtc-audio-processing-dev python3-dev build-essential
pip install webrtc-audio-processing      # обёртка xiongyihui (ReSpeaker-родная)
```
Speex (запасной, проще собрать):
```bash
sudo apt install -y libspeexdsp-dev
pip install speexdsp
```
Примечание: apt-пакет `libwebrtc-audio-processing` обычно v0.3 (старый AEC2 — уже
с нелинейным подавителем, для де-риска хватит). Полноценный **AEC3** — из более
свежего форка webrtc; если v0.3 даёт мало, собрать AEC3 отдельно (PulseAudio
форк `webrtc-audio-processing` v1.x).

---

## Где это в системе
- Девайс-репо: `simonlobgromov/radxa_device` (smart-speaker), `lk.py` —
  туда встраивать real-time AEC между захватом ch1 и STT.
- Wake-word self-trigger решается отдельно **ch5-вето** (AEC для него не нужен).
