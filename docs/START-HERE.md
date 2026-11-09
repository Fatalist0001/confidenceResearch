# START HERE

Этот файл — точка входа для человека и для агента. Он отвечает на три вопроса:
что здесь уже измерено, что можно запускать и чего нельзя делать без явной просьбы.

Всё остальное — подробности:

| файл | что внутри |
|---|---|
| `AGENTS.md` | рабочий контракт: инварианты, структура кода, запреты |
| `README.md` | научная часть: данные, метрики, результаты всех моделей |
| `report.md` | полный отчёт с таблицами проверок и деталями прогонов |
| `RUNS.json` | реестр прогонов: статус, каноничность, числа из артефактов |
| `profiles.json` | флаги и ожидания по семействам моделей |
| `PROJECT_STATE.json` | бинарники, протокол, заблокированные действия |
| `docs/SETUP.md` | как поднять окружение с нуля |
| `docs/LFM25-THINKING-PROTOCOL.md` | отдельный reasoning-протокол LFM2.5-1.2B-Thinking |
| `docs/HOTPOTQA-TRANSFER.md` | замороженный перенос C/13 Qwen 0.8B на HotpotQA без обучения |
| `docs/HOTPOTQA-SOURCE-AUDIT.md` | побайтовая проверка источника и коротких контекстов HotpotQA |
| `docs/history/AGENTS-full-2026-11-02.md` | старый подробный `AGENTS.md` (история) |

---

## 1. Три правила, которые важнее остальных

1. **Ничего не перезаписывать и не удалять.** `pipeline/out/*/hidden_*.npy`,
   `meta.jsonl`, `labels.csv`, `baseline.*`, `probe.*`, `verify.json` — окончательные.
   Новый расчёт — новое имя файла или новый тег. Старые каталоги не чистить «для
   экономии места».
2. **Протокол пробы заморожен.** λ-сетка, splits, точки A/C, выбор по val и один
   замер на test не меняются. Любое изменение — отдельное решение пользователя, а не
   правка по ходу.
3. **Дорогие прогоны — только по явной просьбе.** Сначала пилот 200 строк и гейт,
   потом 6000. Для Kaggle — сначала пилот и подсчёт квоты.

Если сомневаешься — остановись и прочитай `RUNS.json` и `PROJECT_STATE.json`, а не
угадывай по памяти.

---

## 2. Что означают статусы в `RUNS.json`

| поле | значения | смысл |
|---|---|---|
| `status` | `valid` | каталог внутренне согласован, числа можно цитировать |
| | `invalid` | прогон сломан (плохой промпт, старый бинарник, нулевые состояния) — **не цитировать** |
| | `partial` | прогон остановлен, полных артефактов нет — **не цитировать** |
| `canonical` | `true` | именно этот прогон цитируется для данной модели |
| `prompt_regime` | `default_bos` | штатный шаблон, ведущий `<bos>` добавляет токенизатор |
| | `legacy_no_bos` | штатный шаблон, но `<bos>` потерян (прогоны до 2026-11-02) |
| | `no_template_gptoss_final` | `--no-template`, harmony `final` предзаполнен |
| | `no_template_lfm_final` | `--no-template`, cue `Final answer:` |
| | `no_template_lfm_reasoning_final` | replay после 32 reasoning-токенов и принудительного `Final answer:` |
| `probe_status` | `complete` | `probe.json` — результат замороженного протокола |
| | `deferred` | проба отложена намеренно, без новой просьбы не запускать |
| | `not_citable` | прогон невалиден, числа пробы цитировать нельзя |

Важно: `verify.json` проверяет только внутреннюю согласованность каталога. Он **не**
доказывает, что промпт был правильным: `main_gemma4_12b` проходит verify и всё равно
`invalid`.

---

## 3. Какие прогоны цитировать

Канонический прогон — один на модель. Числа ниже сгенерированы из артефактов
(`RUNS.json`), а не переписаны из текста.

| модель | каталог | форма | `filtered_b` | baseline entropy | проба | test AUROC |
|---|---|---|---:|---:|---|---:|
| Qwen3.5-0.8B Q4 | `main_qwen08_q4` | 24×1024 | 0.118 | 0.717 | C / 13 | 0.8709 |
| Qwen3.5-2B Q4 | `main_qwen35_2b_q4` | 24×2048 | 0.253 | 0.794 | C / 12 | 0.8855 |
| Qwen3.5-4B Q4 | `main_qwen35_4b_q4` | 32×2560 | 0.454 | 0.878 | C / 15 | 0.9180 |
| Qwen3.5-9B Q4 | `main_qwen35_9b_q4` | 32×4096 | 0.570 | 0.877 | C / 16 | 0.9172 |
| Gemma 4 E2B | `main_gemma4e2b` | 35×1536 | 0.182 | 0.712 | C / 19 | 0.9351 |
| Gemma 4 E4B | `main_gemma4e4b_bos_master` | 42×2560 | 0.470 | 0.872 | C / 20 | 0.9186 |
| Gemma 4 12B | `main_gemma4_12b_v3` | 48×3840 | 0.572 | 0.826 | C / 28 | 0.9107 |
| Gemma 4 26B-A4B | `main_gemma4_26b_a4b_udq4km` | 30×2816 | 0.708 | 0.843 | C / 16 | 0.9077 |
| gpt-oss-20b | `main_gptoss20b_v2` | 24×2880 | 0.490 | 0.841 | C / 16 | 0.9248 |
| Nemotron 3 Nano 4B | `main_nemotron3nano4b_v2` | 17×3136 | 0.285 | 0.816 | C / 20 | 0.8819 |
| LFM2.5-1.2B Instruct | `main_lfm25_1p2b_instruct_q4` | 16×2048 | 0.177 | 0.765 | C / 9 | 0.8812 |
| LFM2.5-1.2B Thinking | `main_lfm25_1p2b_thinking_r32_q4_v2` | 16×2048 | 0.194 | 0.662 | C / 9 | 0.8813² |
| LFM2.5-2.6B | `main_lfm25_2p6b_q4_final` | 30×2048 | 0.382 | 0.780 | C / 17 | 0.8812 |
| LFM3-1.7B | `main_lfm3_1p7b_qad_q4` | 24×2048 | 0.305¹ | 0.725 | C / 12 | 0.8608¹ |
| YandexGPT 5 Lite 8B | `main_yandexgpt5lite8b_q4` | 32×4096 | 0.568 | 0.771 | C / 16 | 0.9062 |

¹ LFM3: 2509 из 6000 строк обрезаны; метрики только по 3491 завершённой строке.

² Thinking: точка A измерена после 32 reasoning-токенов, у Instruct — до них. По A эти
два прогона не сравниваются; C сопоставима.

Форма — `[6000, число слоёв с l_out, D]`. Для Nemotron в таблице указан реальный номер
слоя (`selected.layer_id`); в `probe.json` поле `selected.layer` — индекс внутри списка
17 доступных слоёв (там 8).

Валидные, но неканонические прогоны, которые нужны для конкретных сравнений:

| каталог | зачем |
|---|---|
| `main_qwen08` | BF16-якорь кванта; сравнивать с `main_qwen08_q4` |
| `main_qwen08_q4_master` | контроль «пин vs master»: те же 6000 строк на новом бинарнике |
| `main_gemma4e4b` | старый прогон без `<bos>`; не сравнивать токен-в-токен с BOS-прогоном |

Невалидные каталоги (`main_gemma4_12b`, `main_gptoss20b_dupinstr`,
`main_nemotron3nano4b`) и остановленный `main_gptoss20b` лежат на диске как улики.
В git они не входят, но и не удаляются.

---

## 4. Замороженный протокол пробы

<a id="frozen-probe-protocol"></a>

Полное описание — `README.md` §6 и `pipeline/probe.py`. Кратко и точно:

| элемент | значение |
|---|---|
| метка | `correct_filtered_b` из `labels.csv` |
| признак | **один** вектор **одного** слоя, точка A или C |
| стандартизация | mean/std, посчитанные **только на train** |
| модель | L2-логистическая регрессия (Newton/IRLS) |
| λ-сетка | `100, 300, 1000, 3000` |
| выбор λ | максимум **среднего val AUROC** по всем конфигурациям точка×слой |
| выбор точки/слоя | максимум **val AUROC** при выбранной λ |
| test | считается **один раз**, после выбора; по test ничего не перевыбирается |
| bootstrap | 1000 пересэмплирований, seed 42 |
| сохранение | `probe_model.npz` (`w`, `mu`, `sd`, JSON `meta`) |

Запрещено: менять λ-сетку, добавлять/убирать точки, выбирать слой по test, объявлять
«победителя» по AUPRC между моделями с разной базовой долей, подгонять протокол под
ориентир 0.86–0.94.

Если каталог старый и в нём нет `probe_model.npz`:

```powershell
& .\pipeline\.venv\Scripts\python.exe pipeline\save_probe_model.py --dir pipeline\out\main_<tag>
```

---

## 5. Что заблокировано

Источник истины — `PROJECT_STATE.json → authorization.blocked_actions` и поле
`blocked` в `profiles.json`. `scripts/run.ps1` откажется запускать эти действия.

| id | что нельзя | почему |
|---|---|---|
| `run:lfm3-sampler-diagnostic` | новый полный прогон LFM3 | сначала дешёвая диагностика сэмплера в `llama-cli` |
| `kaggle:full-run` | полный прогон на Kaggle | квота 30 ч/неделю; сначала пилот и бюджет |
| `protocol:change` | правка λ/splits/точек/метки | ломает сравнимость со всеми прогонами |
| `artifacts:rewrite-or-delete` | перезапись/удаление артефактов | данные дорогие и не воспроизводятся «на лету» |

Разблокировка — только явная просьба пользователя. Технически: повторить команду с
`-ConfirmDeferred`, но **не** делать этого по своей инициативе.

Отдельно: `HALLUCINATION_PLAN.md` и `QUANT_TRANSFER_PLAN.md` — это планы, а не задачи.

---

## 6. Рабочий цикл (локально)

Все команды — из корня репозитория.

```powershell
# 1. окружение и бинарники
powershell -File scripts\bootstrap.ps1          # на чистом клоне: исходники + venv
powershell -File scripts\run.ps1 -Stage doctor -Profile qwen -Model D:\models\x.gguf

# 2. вход и промпт (без загрузки весов)
powershell -File scripts\run.ps1 -Stage inputs -Profile qwen -Tag qwen4b_v2
powershell -File scripts\run.ps1 -Stage prompt -Profile qwen -Tag qwen4b_v2 -Model D:\models\x.gguf

# 3. пилот 200 -> гейт -> полный прогон 6000
powershell -File scripts\run.ps1 -Stage pilot -Profile qwen -Tag qwen4b_v2 -Model D:\models\x.gguf
powershell -File scripts\run.ps1 -Stage main  -Profile qwen -Tag qwen4b_v2 -Model D:\models\x.gguf

# 4. разметка, проверка, baseline, проба
powershell -File scripts\run.ps1 -Stage label    -Profile qwen -Tag qwen4b_v2
powershell -File scripts\run.ps1 -Stage verify   -Profile qwen -Tag qwen4b_v2
powershell -File scripts\run.ps1 -Stage baseline -Profile qwen -Tag qwen4b_v2
powershell -File scripts\run.ps1 -Stage probe    -Profile qwen -Tag qwen4b_v2

# 5. контроль детерминизма и сравнение
powershell -File scripts\run.ps1 -Stage probe2 -Profile qwen -Tag qwen4b_v2
powershell -File scripts\run.ps1 -Stage compare -Profile qwen -Tag qwen4b_v2 `
  -Ref pipeline\out\main_qwen35_4b_q4\probe_scores.csv -Split test
```

Что раннер проверяет сам:

- `-Tag` не должен совпадать с существующим каталогом или логом;
- `pilot/main` требуют успешную запись `prompt_<tag>.<stage>.json`; hash входа,
  профиль, модель и binary commit/build должны совпасть;
- перед prompt/generation фактический `--version` бинарника сверяется с
  `PROJECT_STATE.json`;
- `-Stage main` требует `pilot_<tag>/gate.json` с `passed=true` и тот же `-Model`;
- pilot автоматически получает `labels.csv` и повторный class-balance gate; отдельный
  `-Stage label` относится только к main; hash исходных вопросов сверяется с
  `runner.json` перед разметкой;
- `-Stage probe` требует `verify.json` с `fails=0`;
- `-Stage prompt` сверяет `add_bos` и первый токен с `profiles.json`;
- сборщик промпта должен быть идемпотентен (побайтово);
- логи пишутся в файл, без `Tee-Object`;
- `-b/-ub` не передаются никогда.

Проверка документации и реестра:

```powershell
& .\pipeline\.venv\Scripts\python.exe scripts\build_runs.py --check
& .\pipeline\.venv\Scripts\python.exe scripts\check_state.py
& .\pipeline\.venv\Scripts\python.exe -m unittest discover -s pipeline\tests -v
```

После нового прогона `python scripts\build_runs.py` пересобирает числовые поля
`RUNS.json` прямо из `config/meta/labels/baseline/probe/verify`; руками редактируются
только статусы и provenance-каталог в самом генераторе.

---

## 7. Профили

| профиль | когда |
|---|---|
| `standard` | любая модель, для которой нет отдельного профиля |
| `qwen` | Qwen3.5 0.8B / 2B / 4B, Q4_K_M |
| `qwen-hotpotqa` | только Qwen3.5-0.8B Q4_K_M, сохранённая C/13 проба, 200+500 HotpotQA; без обучения/калибровки |
| `qwen9b` | Qwen3.5 9B: только `-ngl 99`, лог в файл |
| `gemma` | Gemma 4 E2B / E4B, ожидается `add_bos=1`, первый токен `2` |
| `yandex` | YandexGPT 5 Lite 8B |
| `nemotron` | Nemotron 3 Nano 4B: 17 из 42 слоёв, `verify` даёт `warns=1` |
| `gptoss20b` | gpt-oss-20b: готовый harmony `final`, `--no-template`, `-ngl 11`, `--batch-rows 8` |
| `lfm25-2.6b` | LFM2.5-2.6B: готовый cue `Final answer:`, `--no-template` |
| `lfm25-1.2b-instruct` | LFM2.5-1.2B Instruct |
| `lfm25-1.2b-thinking` | reasoning 32 + final 64 + replay; см. отдельный протокол |
| `lfm3` | LFM3-1.7B: диагностический sampler-профиль, гейт 10%; новый полный прогон заблокирован |
| `gemma4-12b-kaggle` | только Kaggle, локально не запускается |
| `gemma4-26b-a4b-kaggle` | только Kaggle, 2×T4; локально не запускается |

Флаги берутся из `profiles.json`, а не из памяти. Если профиля нет — `standard`, но
тогда `prompt`-стадия обязательна.

При ручном `-InputFile` всегда передавайте `-QuestionsFile` с исходными,
неотрендеренными вопросами: `label.py` использует их для `filtered_a/b`.

---

## 8. Рецепты

### Новая модель того же семейства (например, Qwen 3.5 4B другого кванта)

1. `-Stage doctor -Profile qwen -Model <gguf>`
2. `-Stage prompt -Profile qwen -Tag <new> -Model <gguf>` — глазами проверить дамп.
3. `-Stage pilot -Profile qwen -Tag <new> -Model <gguf>` — гейт должен пройти.
4. Прочитать 10 ответов из `pilot_<new>/meta.jsonl`, включая обрезанные.
5. `-Stage main`, затем `label → verify → baseline → probe → probe2`.
6. Записать README каталога, добавить provenance-строку в `scripts/build_runs.py` и
   выполнить `python scripts/build_runs.py` (числа он возьмёт из артефактов сам).

### Незнакомое семейство

Начать с `-Profile standard`. Обязательно:

- `hs-prompt-dump` показывает `add_bos` и первый токен — сверить с тем, что ожидает
  шаблон;
- в промпте ровно одна инструкция `Answer with just the answer, no explanation.`;
- пилот 200 проходит гейт; если почти всё `truncated` — не запускать main, а разобраться
  с шаблоном/бюджетом.

### gpt-oss-20b

```powershell
powershell -File scripts\run.ps1 -Stage prompt -Profile gptoss20b -Tag gptoss_v3 -Model <gguf>
```

Раннер соберёт `data/{pilot,main}_gptoss_final.jsonl` через `make_gptoss_prompt.py`,
проверит идемпотентность и запустит с `--no-template`, `-ngl 11`, `--batch-rows 8`.
Инструкция в готовом промпте должна встречаться ровно один раз.

### LFM2.5-2.6B

Профиль `lfm25-2.6b` соберёт вход с cue `Final answer:` и запустит `--no-template`.
Штатный шаблон открывает ` thinking`, поэтому его использовать нельзя.

### LFM2.5-1.2B-Thinking

Этот режим состоит из reasoning/final preprocessing и replay состояний. Бюджеты,
stop-rule и точки зафиксированы в `docs/LFM25-THINKING-PROTOCOL.md`.

```powershell
$MODEL = 'D:\Lm studio\.lmstudio\models\LiquidAI\LFM2.5-1.2B-Thinking-GGUF\LFM2.5-1.2B-Thinking-Q4_0.gguf'
$TAG = 'lfm25_1p2b_thinking_r32_q4_v2'

powershell -File scripts\prepare_lfm_thinking.ps1 -Stage pilot -Model $MODEL -Tag $TAG
powershell -File scripts\run.ps1 -Stage prompt -PromptKind pilot `
  -Profile lfm25-1.2b-thinking -Tag $TAG -Model $MODEL
powershell -File scripts\run.ps1 -Stage pilot -Profile lfm25-1.2b-thinking `
  -Tag $TAG -Model $MODEL -QuestionsFile data\pilot.jsonl

# Только после passed gate и просмотра ответов:
powershell -File scripts\prepare_lfm_thinking.ps1 -Stage main -Model $MODEL -Tag $TAG
powershell -File scripts\run.ps1 -Stage prompt -PromptKind main `
  -Profile lfm25-1.2b-thinking -Tag $TAG -Model $MODEL
powershell -File scripts\run.ps1 -Stage main -Profile lfm25-1.2b-thinking `
  -Tag $TAG -Model $MODEL -QuestionsFile data\main.jsonl
powershell -File scripts\run.ps1 -Stage label -Profile lfm25-1.2b-thinking `
  -Tag $TAG -QuestionsFile data\main.jsonl
```

Дальше обычные `verify → baseline → probe → probe2`:

```powershell
powershell -File scripts\run.ps1 -Stage verify -Profile lfm25-1.2b-thinking -Tag $TAG
powershell -File scripts\run.ps1 -Stage baseline -Profile lfm25-1.2b-thinking -Tag $TAG
powershell -File scripts\run.ps1 -Stage probe -Profile lfm25-1.2b-thinking -Tag $TAG
powershell -File scripts\run.ps1 -Stage probe2 -Profile lfm25-1.2b-thinking -Tag $TAG
powershell -File scripts\run.ps1 -Stage compare -Profile lfm25-1.2b-thinking -Tag $TAG `
  -Ref out\main_lfm25_1p2b_instruct_q4\probe_scores.csv -Split test
```

`meta.raw.jsonl` сохраняет неизменённый replay-meta; итоговый `meta.jsonl` получает
baseline из final-generation pass до запуска gate. Если reasoning-проход прерван,
`prepare_lfm_thinking.ps1 -Stage main -Resume` продолжит с последней завершённой строки
в новый каталог и сольёт части, не меняя исходный.

Ориентир завершённого прогона: 6000/6000, 5961 размеченная строка, probe λ=1000, C/9,
test AUROC 0.8813 [0.855,0.905]; против Instruct ΔAUROC +0.0016 [−0.0278,+0.0316].

### Посмотреть существующий прогон

```powershell
& .\pipeline\.venv\Scripts\python.exe `
  llama.cpp-hiddenstates\examples\hidden-states\load_example.py `
  pipeline\out\main_qwen35_4b_q4 --k 3
```

Числа брать из `probe.json` / `baseline.json` / `verify.json`, а не из README: README
может отставать, `RUNS.json` генерируется из артефактов.

### HotpotQA: замороженная проба, а не новое обучение

Протокол — `docs/HOTPOTQA-TRANSFER.md`. Разрешённый предварительный тест использует
только Qwen3.5-0.8B Q4_K_M и готовую пробу `main_qwen08_q4/probe_model.npz`
(C / model layer 13, λ=1000). В целевом массиве сохранён один слой, его index равен 0.

В `data/hotpotqa_qwen08_v2/protocol.json` до генерации зафиксированы 200 disjoint
pilot ID, 500 evaluation ID, входы, эталоны, token diagnostics и контрольные суммы
модели, пробы, runtime и кода. Все фактически предоставленные документы остаются в
исходном порядке. У 2 pilot и 7 evaluation вопросов в оригинальном JSON менее 10
абзацев; пользователь разрешил сохранить эти вопросы и весь доступный текст после
проверки источника. Дополнять контекст или заменять ID нельзя.

Workflow: `doctor -> inputs -> prompt -> pilot`, затем passed gate + full state
verify + просмотр ответов, после чего `main -> label -> verify -> transfer`.
Для каждого этапа используйте `scripts/run.ps1 -Profile qwen-hotpotqa -Tag <new>`;
для генерации нужен исходный GGUF Qwen 0.8B Q4_K_M. `-Stage all -DryRun` показывает
этот маршрут без запуска модели. Повтор требует нового тега и явной просьбы.

**Нельзя запускать `baseline`, `probe` или `probe2` на transfer-каталоге:** эти
стадии обучают или калибруют на целевых данных. Раннер отказывает даже если профиль
при анализе опущен либо подставлен другой. Сохранённые `w/mu/sd`, точка, слой и λ
не меняются.

Оценка HotpotQA использует официальный whole-answer `hotpot_em` и answer F1, а не
тренировочную TriviaQA-метку `correct_filtered_b`. Все попытки входят в accuracy;
probe/logprob/negative entropy сравниваются на одной primary-выборке завершённых
непустых ответов с валидными состояниями и first-token scores. Выходы —
`transfer.json` и `transfer_scores.csv`, оба create-only. Результаты в реестре
отделены полем `evaluation` и `probe_status=frozen_transfer` от новых обученных проб.

### Kaggle

Локальный раннер Kaggle не управляет. Порядок:

```powershell
powershell -File kaggle\gemma4_12b\push.ps1
powershell -File kaggle\watch.ps1 -Kernel odinodinodin/hs-gemma4-12b -IntervalSec 60

powershell -File kaggle\gemma4_26b_a4b\push.ps1
powershell -File kaggle\watch.ps1 -Kernel odinodinodin/hs-gemma4-26b-a4b -IntervalSec 60
```

Перед запуском и после — `kaggle quota`. Полный прогон — только после пилота и подсчёта
бюджета; см. `kaggle/README.md`.

---

## 9. Если что-то пошло не так

| симптом | вероятная причина | что делать |
|---|---|---|
| гейт пилота: `truncated` выше лимита | ответ не влезает в 64 токена | проверить шаблон/`thinking`, обсудить бюджет; main не запускать |
| гейт: `chat-protocol leak` | модель печатает `<|channel>`, `<channel|>`, `thought` | промпт/шаблон не тот; не запускать main |
| гейт: `answers are looping` | вырожденная генерация | проверить квант и шаблон, повторить пилот |
| `add_bos=1`, но первый токен не BOS | потерян `<bos>` (случай Gemma 4 12B) | остановиться, смотреть `prompt_<tag>.*.txt`; не запускать |
| `verify_run.py`: нулевой слой | старый бинарник или `l_out` не на всех слоях | проверить `--version`, список `layers`; для Nemotron норм. 17 слоёв |
| `verify_run.py`: дыры в `row` | прогон прерван | не «чинить» файл; перезапускать в новый тег |
| `probe.py`: `no usable rows for split` | нет test/val или метки пустые | проверить `labels.csv` и splits |
| `probe2` не совпал побайтово | недетерминизм | не перезаписывать оригинал; разбираться отдельно |
| `main` отказывается стартовать | нет пройденного pilot gate или другой `-Model` | сначала пилот тем же тегом и моделью |
| `apply_probe_model.py`: `target D != probe D` | другая размерность | перенос весов невозможен без выравнивания |

Общий принцип: **нельзя «исправлять» артефакт.** Новый расчёт — новый каталог/файл,
старый остаётся как есть.

---

## 10. Где лежат числа

| файл | что читать |
|---|---|
| `pipeline/out/main_<tag>/probe.json` | `selected_lambda`, `selected`, `metrics.val/test`, `controls` |
| `probe_scores.csv` | скоры по строкам (val + test) для сравнений |
| `probe_curve.csv` | вся сетка точка×слой×λ |
| `probe_model.npz` | веса, `mu`, `sd`, JSON `meta` |
| `baseline.json` | AUROC/AUPRC logprob и энтропии по splits |
| `labels.csv` | построчные метки и split |
| `verify.json` | `fails`, `warns`, `checks` |
| `gate.json` | `passed`, критерии, оценка времени полного прогона |
| `runner.json` | фактические флаги, модель, бинарник, commit/build |
