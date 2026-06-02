# Обучение openWakeWord на слове «акылай» — история и план

> Рабочий журнал. Цель: обучить **openWakeWord** (этот репозиторий) на кыргызском
> wake word **«акылай»** на CPU, получить `akylai.onnx`, который хорошо работает,
> когда звук идёт **через колонку** (устройство). Это пилот на подвыборке; потом масштаб.

## ⏭ ТЕКУЩИЙ СТАТУС (обновлено 2026-05-29)

**Лучшая модель: `akylai_real.onnx`** — первая модель с ЖИВЫМ голосом в обучении. На честном
held-out тесте живых голосов **recall 98% @ 0.5** (cons 88%, w10 95%). Запушена в публичный
репо `git@github.com:thezholdoshbekov/openwakeword_akylai.git` (коммит «first commit»).

**Что сделано 2026-05-29:**
1. **Добавили реальные позитивы.** Скачали gated-датасет `Simonlob/Wake_word_Akylai`
   (`hf_download.py`, токен из `$env:HF_TOKEN`): 667+58 живых «акылай» + 761+101 живых негатива,
   ~1 с, 16 кГц. До этого позитивы были **только TTS** (~24k) — отсюда ложные на реальной речи.
2. `add_real_positives.py`: влили 667 живых позитивов **×10** в `positive_train` (апвейт против 24k TTS;
   каждая копия получает свою аугментацию), 761 негатив в `negative_train`, тестовую часть (58/101)
   отложили в `positive_real_test`/`negative_real_test` (честный held-out на живых).
3. Пересчитали train-фичи (`augment_features.py --train-only`, Conservative-рецепт):
   **31114 позитивов / 65876 негативов** (16,96).
4. Дообучили → `akylai_real.onnx` (= скопирована из `akylai.onnx`).
5. Оценка: `eval_real.py` (живые), `eval_recall.py` (непрерывные ww_recall_2026/2027), `eval_det.py` (DET).
6. Деплой: `device_detect.py` (live-инференс на колонке) + модель в `training/akylai/models/` → push.

**Результаты (порог → recall / ложные):**
| модель | живые поз @0.5 | живые поз @0.9 | ложные(конфуз) @0.9 | непрерыв. события (2026/2027) |
|---|---|---|---|---|
| **real** | **98.3%** | 93.1% | 9.9% | 30 / 51 |
| cons (TTS) | 87.9% | 77.6% | 4.0% | 24 / 35 |
| w10 | 94.8% | 84.5% | 11.9% | 28 / 42 |

→ Живые позитивы дали **+10 пунктов recall** на реальной речи. Платим ростом ложных на конфузерах
(4%→10% @0.9) — выгодный размен; зажимается порогом ~0.93. На непрерывных записях real даёт больше
событий (это recall-тесты, человек повторяет «акылай»; livekit-модель там слабая, много пропусков).

**НЕ доделано / следующие рычаги:**
- Прослушать «лишние» события на непрерывных записях (`D:/akylai_oww/listen_real/disputed/`) — recall или ложные?
- Выбрать рабочий порог (старт ~0.9–0.93).
- Зажать ложные на конфузерах: таргетные негативы кулай/алтынай/ылай.
- Запустить вживую на колонке (RK3566) через `device_detect.py`, замерить реальные скоры.
- tflite для устройства (сейчас конвертация падает — нет `onnx_tf`).

Запуск из корня репо `C:\Users\THEZHOLDOSHBEKOV\openWakeWord`; для `-m openwakeword.train` PYTHONPATH не нужен,
для `training/akylai/*.py` и `examples/*.py` — нужен (`$env:PYTHONPATH="C:\Users\THEZHOLDOSHBEKOV\openWakeWord"`).

## Рецепт аугментации (Conservative — текущий, в `augment_features.py`)

Применяется к ОБОИМ классам, порядок: **device-color → RIR колонки → фон+MUSAN → pitch → AGC**.
Заменяет встроенный `train.py --augment_clips`, чтобы управлять вероятностями И добавить
device-специфику (из анализа 20 клипов с колонки, см. `device_analysis/`).

| Шаг | Вероятность | Параметры | Зачем |
|---|---|---|---|
| **device-color** (low-pass + mid peak-EQ) | 0.40 | LP 3.5–4.5 кГц (ord 2–4); mid +3..6 дБ @2–3.5 кГц, Q 0.7–1.5 | колонка режет верх (rolloff ~1.5к против 2.5к) и горбит 2–3.5 кГц |
| **RIR колонки** | 0.40 (на батч) | свёртка с импульсом колонки | у колонки резонанс/реверберация (flatness ×2.5 ниже) |
| **фон + MUSAN** | 0.60 (на батч) | SNR +5..20 дБ | устойчивость к реальному шуму |
| **pitch** | 0.25 (на батч) | ±3 полутона | разнообразие голосов |
| **AGC/AEC** | 0.0 (OFF) | — | в Conservative отключено |

**История рецепта:** сначала был **Balanced** (жёстче: device .60, SNR −5..15, RIR .70, AGC .30) —
давал AUT ~0.04, но recall@(≤1 ложн/час) всего ~40%. Гипотеза «жёсткая аугментация вредит»
**подтвердилась частично**: Conservative поднял recall@оп.точке до ~58%. Но **главный скачок** дали
не параметры аугментации, а **живые позитивы** (recall 88%→98%). Урок: данные > подкрутка аугментации.


## Контекст и решения

- **Пайплайн:** именно openWakeWord (`openwakeword/train.py`), НЕ livekit_ww_akylai. (Решение пользователя.)
- **Железо:** машина только CPU (`torch 2.2.0+cpu`, без CUDA).
- **Данные:** приватный HF `kyrgyz-ai/AkylAi_Wake_Word_V4_Dataset` (токен уже сохранён локально).
- **Генерацию синтетики пропускаем** (`--generate_clips` НЕ запускаем): её англ. генератор негативов
  бесполезен для кыргызского. Подаём реальные клипы из датасета.
- **Вывод только ONNX** (без `--convert_to_tflite`: TF/onnx_tf не ставятся на Win/py3.12).
- **Пилот** на subset-манифестах (`ww_train_subset` / `ww_val_subset`).

## Окружение (установлено)

```
pip install torchinfo speechbrain audiomentations==0.37.0 torch-audiomentations acoustics mutagen pronouncing
pip install numpy==1.26.4 scipy==1.12.0   # torch 2.2 требует numpy<2; acoustics требует scipy<1.13 (sph_harm)
```
Проверено: `import openwakeword.train` работает без ошибок.

## Созданные файлы (`training/akylai/`)

- **prepare_data.py** — конвертирует parquet → папки WAV, которые ждёт `train.py`:
  - `oww/akylai/{positive_train,positive_test,negative_train,negative_test}/clip_*.wav` (моно 16 кГц 16-бит).
  - Позитивы пересчитываются 22050→16000.
  - **Аугментация тут:** каждый позитив train дублируется со сдвигом тона **±4 полутона** (голос выше/ниже).
  - Все 2949 «похожих слов» (confusables) → негативы (90/10 train/test).
  - Распаковка фонов: подкасты → `oww/backgrounds_wav/`, MUSAN → `oww/musan_wav/`.
- **akylai.yml** — конфиг обучения: `model_type: dnn`, `layer_size: 32`, `steps: 8000` (пилот),
  `background_paths: [backgrounds_wav, musan_wav]`, `rir_paths: [rirs_device]`,
  `false_positive_validation_data_path` → скачанный `validation_set_features.npy`,
  `feature_data_files: {}`, `batch_n_per_class: {positive:50, adversarial_negative:50}`.
- **_stub/generate_samples.py** — заглушка-ничего, чтобы `train.py` мог сделать
  `from generate_samples import …` (он импортирует Piper безусловно), не клонируя piper-sample-generator.

## Что уже подготовлено (на диске)

| Что | Путь | Кол-во |
|---|---|---|
| positive_train (вкл. ±4 полутона) | `oww/akylai/positive_train` | 9000 |
| negative_train (база + confusables) | `oww/akylai/negative_train` | 9543 |
| positive_test | `oww/akylai/positive_test` | 2736 |
| negative_test | `oww/akylai/negative_test` | 1058 |
| фоны подкастов | `oww/backgrounds_wav` | 3000 |
| MUSAN шум/музыка | `oww/musan_wav` | 3000 |
| RIR колонки | `oww/rirs_device/rir_device.wav` | 1 (0.2с, 16к) |
| FP-валидация | `training/akylai/validation_set_features.npy` | (481345, 96) |

Датасет скачан в `training/akylai/data/` (positives 3.3G, confusables 80M, musan 768M, backgrounds 1.4G, manifests).

## Аугментации

openWakeWord сам применяет на лету (`openwakeword/data.py::augment_clips`):
EQ(0.25), TanhDistortion(0.25), PitchShift±3(0.25), BandStopFilter(0.25), AddColoredNoise(0.25),
**AddBackgroundNoise(0.75, SNR −10..+15 дБ)**, Gain(1.0), **RIR(0.5, на весь батч)**.

Наши добавления:
- **MUSAN** (шум/музыка) добавлен в пул фонов.
- **Сдвиг тона ±4** у позитивов (предварительно, в prepare_data.py).
- **RIR колонки** (`rir_device.wav`) — «звучание колонки». Свёртка воспроизводит частотную
  характеристику устройства (центроид ~700 Гц против ~1300 Гц у корпуса — приглушённо + резонанс,
  см. `livekit_ww_akylai/spectral_search/stats.txt`).

**ВАЖНО (открытый вопрос):** `train.py` НЕ даёт менять вероятности аугментаций (они зашиты).
Чтобы управлять **пропорцией RIR колонки** и силой фона, нужен отдельный скрипт
`augment_features.py` (повторяет блок `--augment_clips` из train.py, но с настраиваемыми
`augmentation_probabilities` и нашим RIR), затем `train.py --train_model` читает готовые `.npy`.

## Анализ 20 записей с устройства (2026-05-28)

20 клипов с реального целевого железа (`device_clips/clip_01..20.wav`, из livekit-репо).
Формат — **6 каналов, 16 кГц, по 5 с** (микрофонная решётка типа ReSpeaker):
- **ch0** — обработанный/бимформленный канал (RMS ≈ −28 dBFS после AEC+AGC) → **это вход модели на устройстве**, его и анализируем;
- ch1–ch4 — сырые микрофоны (≈ −38 dBFS); ch5 — пустой loopback.

В каждом клипе **по 2 произнесения «акылай»** (~0.65 с, пауза ~1 с) → фактически ~40 позитивов.

Скрипт: `analyze_device_clips.py` → `device_analysis/{stats.txt,segments.txt,fig1_overview.png,fig2_features.png,fig3_spectral.png}`.

**Домен-гэп ch0 vs train (60 случайных позитивов):**

| Метрика | TRAIN | DEVICE ch0 |
|---|---|---|
| Центроид, Гц | 1259 ±232 | **893 ±40** |
| Rolloff85, Гц | 2567 ±539 | **1548 ±101** |
| Flatness | 0.010 | **0.004** |
| Средний JSD | — | **0.699** |

**Форма окраски устройства (главное):**
- до ~1.5 кГц — почти как train;
- **2–3.5 кГц — устройство +5…+13 дБ** (presence-горб/резонанс);
- **выше 4 кГц — устройство −10…−15 дБ** (резкий завал верхов).

Разброс устройства крошечный (±40 против ±232) → это **узкий плотный кластер на «тусклом» краю** train-распределения. Проблема = сдвиг, а не разброс. ch0 (JSD 0.70) ближе к train, чем сырой мик/AEC из прошлых сессий (0.76–0.81): обработка колонки частично компенсирует.

**Выводы для `augment_features.py`:**
1. **HF-завал обязателен** (LP/полочный срез выше ~3–4 кГц) — главный инструмент.
2. **Mid-boost 2–3.5 кГц** — добавить peak-EQ в этой полосе на часть клипов (повторить «горб»).
3. **RIR колонки** — высокая вероятность (резонанс, flatness ×2.5).
4. Цель — **сдвинуть** train-кластер к устройству, а не «размазать» шумом.

## Аугментация фич: `augment_features.py` (рецепт Balanced, 2026-05-28)

Свой скрипт вместо штатного `--augment_clips` — чтобы управлять вероятностями И добавить
device-аугментации из анализа. Выдаёт **те же 4 файла фич**, что ждёт `train.py --train_model`:
`oww/akylai/{positive,negative}_features_{train,test}.npy` (форма `(N, 16, 96)`).

Рецепт (параметрический, к обоим классам), порядок: **device-color → RIR → фон → pitch → AGC**.
Все числа — в блоке `PARAMS` в начале скрипта (там и крутить):
- **device-color** p=0.60: low-pass (срез 3–4 кГц, порядок 2–4) + mid-boost peak-EQ (+5…9 дБ, 2–3.5 кГц, Q 0.7–1.5).
- **RIR колонки** p=0.70 (свёртка с `rirs_device/rir_device.wav`).
- **фон+MUSAN** p=0.75 (SNR −5…+15 дБ), **pitch ±3** p=0.25.
- **AGC** p=0.30 (глоб. −5…−8 дБ + лёгкий доп. LP 4.5–6 кГц).

Зачем что — см. раздел «Анализ 20 записей» и `device_analysis/stats.txt`.

**Важно про запуск (Windows):**
- Один раз скачать feature-модели: `python -c "import openwakeword.utils as u; u.download_models([])"`
  (создаёт `openwakeword/resources/models/{melspectrogram,embedding_model}.onnx`).
- В скрипте есть **патч `trim_mmap`**: штатный падает на Windows (`os.remove` при открытом memmap) —
  наша версия закрывает memmap перед удалением. Апстрим не трогаем.
- **Конфиг править НЕ нужно**: скрипт пишет ровно те имена файлов, что читает `train.py --train_model`.
- Dry-run: `python training/akylai/augment_features.py --subset 64`. Полный прогон — самый долгий шаг на CPU.

Новый порядок обучения:
```bash
python training/akylai/augment_features.py                                            # наши фичи (долго)
python openwakeword/train.py --training_config training/akylai/akylai.yml --train_model   # БЕЗ --augment_clips
```

## Результаты обучения (полный объём, 2026-05-28)

Данные на **D:** (`D:/akylai_oww/`, C: был забит): 24 444 поз + 65 115 нег (вкл. 44K подкастов,
MUSAN, confusables, **365 реальных user_voice-негативов**) → 4 файла фич `(N,16,96)`.
Модель: `D:/akylai_oww/oww/akylai.onnx`.

**Критический баг openWakeWord `auto_train`:** `best_val_fp` инициализируется в 1000 и НИКОГДА не
обновляется → `best_val_fp > target` всегда True → вес негативов **удваивается** после каждой из 3 фаз.
Плюс мы в другом режиме: дефолт 1500 рассчитан на пул лёгких `ACAV100M` негативов (`batch_n_per_class`
ACAV100M:1024), которого у нас НЕТ — у нас 50/50 батчи **тяжёлых** негативов.

**Прогон 1** (`max_negative_weight: 1500`, эскалация → ~6000): модель **схлопнулась** —
recall на in-dist тесте 2%, на device_clips 0/20. Негативы давили позитивы.

**Фикс:** `max_negative_weight: 10`, `target_false_positives_per_hour: 1000000` (глушит эскалацию).

**Прогон 2** (weight=10):
- in-dist тест: POS 76.6% @0.5 (median 0.98), NEG 3.8% @0.5;
- **device_clips: 20/20 = 100%** (peak 0.95–0.998) — device-color аугментация работает!
- benchmark позитивы (4/4) ловятся 0.98+;
- **НО confusables ложно срабатывают**: benchmark `алтынай/асылай/бурулай/николай/чолпонай` 0.92–0.996;
  user_voice `лайк 0.94, кулай 0.95, Алтынай 0.995, Мамай 0.96`. Перелёт в сторону recall.

**Вывод:** оптимум `max_negative_weight` между 10 и 1500 (эскалация ВЫКЛ). След. шаг — попробовать
~100–300, цель: держать device recall ~100% и опустить confusables (-ай/-лай) ниже порога.
Скрипты: `eval_model.py` (recall/FP на device_clips + benchmark + user_voice).

## Sweep веса негативов (2026-05-28)

Бэкапы моделей в `D:/akylai_oww/oww/`: `akylai_w10.onnx`, `akylai_w150.onnx`. Eval: `eval_compare.py`.

| Метрика | w10 | w150 |
|---|---|---|
| device recall (ch0) | **20/20** (mean .98) | 9/20 (mean .46) |
| in-dist POS>0.5 | 76.6% | 46.9% |
| in-dist NEG>0.5 | 3.8% | 0.2% |
| лайк / ылай / асылай / николай | все FP (.88–.96) | все ✅ (.14–.33) |
| **кулай** | .95 FP | **.67 FP** |
| **алтынай** | .99 FP | **.82 FP** |

**Выводы:**
- `weight=10` слишком мягко (recall 100%, но 6/6 confusables ложные); `weight=150` слишком жёстко
  (recall рухнул 20→9), хотя 4/6 confusables починил.
- **«кулай» и «алтынай» пробивают при любом весе** — фонетически слишком близки к «акылай».
  Весом их не убрать, не угробив recall. Нужен **таргетный фикс данными**: oversample именно этих
  hard-confusables (отдельный feature-файл + слот в `batch_n_per_class`), либо больше реальных записей
  «кулай/алтынай» (это и есть Phase 3 «real-voice calibration» из карточки датасета).
- Вживую (микрофон ноута) w10 ловит «акылай» уверенно; FP на похожих слышны — совпадает с eval.

**Следующий шаг:** либо weight ~40–60 (recall ~90% + 2 стойких FP — компромисс), либо таргетный
oversample кулай/алтынай (правильнее). Аугментация (device-color) — подтверждена, не трогаем.

## Настоящий тест (DET/AUT/FPPH, как в livekit) — 2026-05-28

Скрипт `eval_det.py` (повторяет `livekit_ww_akylai/.../eval/evaluate.py`, автономно): DET-кривая,
AUT (площадь под DET, меньше=лучше), FPPH (ложные/час на 11.3 ч general-валидации), recall@опт.порог.
Выход: `D:/akylai_oww/eval/det_compare.png`.

```
model        AUT    rec@.5  fpph@.5  optThr  rec@opt  fpph@opt  device
w10          0.045   76.9%   28.32    0.99    36.0%     0.71    20/20
w150         0.041   47.6%    1.59    0.60    43.1%     0.97     9/20
```

**КЛЮЧЕВОЙ вывод: w10 и w150 лежат на ОДНОЙ DET-кривой** (AUT почти равны) — вес негативов лишь
двигает рабочую точку, НЕ улучшает модель. Дальше крутить вес бессмысленно.

**Честное состояние:** на чистом «акылай» (device) recall 100%, но при ≤1 ложного/час на реальной речи
**recall ~40%** → НЕ продакшн. Ориентир A4 из датасета: ~99% recall @ <1 FA/h. Разрыв большой.

**Чтобы улучшить (опустить AUT), а НЕ двигать точку:**
1. **Реальные позитивы «акылай» в train** (сейчас 0 — всё TTS-синтетика). Самый сильный рычаг.
2. Таргетные данные на кулай/алтынай (не вес).
3. Больше обучения / крупнее модель (layer_size) / сильнее/точнее аугментация.

Бэкапы: `akylai_w10.onnx` (лучший recall), `akylai_w150.onnx`, текущий `akylai.onnx`=w150.

## Что осталось сделать (TODO)

- [x] **Глубокий акустический анализ** записей с устройства — сделано на 20 ch0-клипах,
  см. раздел выше и `device_analysis/`. (RIR/реверберацию по sweep см. ещё `analyze_device.py`
  + `livekit_ww_akylai/spectral_search/`.)
- [x] **`augment_features.py`** с настраиваемыми пропорциями (рецепт Balanced) — написан, dry-run ок.
- [ ] **Полный прогон аугментации** → 4 `.npy` (на всех клипах).
- [ ] **Обучение** `train.py --train_model` → `akylai.onnx`.
- [ ] Написать `augment_features.py` с настраиваемыми пропорциями (RIR колонки высокая) → 4 файла `.npy`.
- [ ] Обучение: `python openwakeword/train.py --training_config training/akylai/akylai.yml --train_model` → `akylai.onnx`.
- [ ] Проверка: реагирует на «акылай», молчит на похожих (`лайк`, `кулай`, `ылай`, `Алтынай`),
  тест на `ww_session*.wav`.
- [ ] Полный прогон: полные манифесты, больше шагов, возможно GPU.

## Как запускать

```bash
# из корня репозитория c:\Users\THEZHOLDOSHBEKOV\openWakeWord
python training/akylai/prepare_data.py             # подготовка данных (уже сделано)
python training/akylai/augment_features.py         # НАШИ фичи с device-аугментацией (долго)
python openwakeword/train.py --training_config training/akylai/akylai.yml --train_model   # БЕЗ --augment_clips
# тест микрофоном:
python examples/detect_from_microphone.py --model_path training/akylai/oww/akylai.onnx --inference_framework onnx
```
(Штатный путь `--augment_clips` тоже работает, но не даёт менять вероятности и device-color — поэтому используем свой скрипт.)

## Найденные факты / риски

- Звук колонки сильно приглушён (LP ~700 Гц) и резонансный → RIR колонки это и накладывает.
- Реальные сессии: quiet / crowded / female — для проверки и для оценки реверберации.
- Установка тянет numpy 2 (ломает torch) и новый audiomentations/scipy — поэтому версии запинены (см. выше).
- На CPU извлечение признаков — самый долгий шаг; пилот держим маленьким.
