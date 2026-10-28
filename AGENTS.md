# AGENTS.md

Проект: CLI-инструмент `llama-hidden-states` — извлечение скрытых состояний всех слоёв
GGUF-модели (позиции A и C) поверх llama.cpp. Один одиночный режим + пакетный режим
для массовых прогонов.

## Где живёт код (важно)

Проект **самодостаточен**: репозиторий инициализирован в корне каталога, сюда входят
документация и полная копия llama.cpp **с нашим примером**:

- удалённый репозиторий: <https://github.com/Fatalist0001/confidenceResearch> (private, ветка `main`, `origin`)
- апстрим llama.cpp в копии: коммит `d4cff114c`; исходный чекаут `D:\llama.cpp` — чистый апстрим без наших правок

| Путь | Что это |
|---|---|
| `llama.cpp-hiddenstates\` | полная копия llama.cpp (коммит `d4cff114c`) **с нашим примером** |
| `llama.cpp-hiddenstates\examples\hidden-states\hidden-states.cpp` | вся программа (~42 КБ) |
| `llama.cpp-hiddenstates\examples\hidden-states\CMakeLists.txt` | таргет `llama-hidden-states` |
| `llama.cpp-hiddenstates\examples\hidden-states\load_example.py` | просмотр результатов |
| `llama.cpp-hiddenstates\examples\CMakeLists.txt` | **изменён на одну строку**: `add_subdirectory(hidden-states)` |
| `report.md` | полный отчёт: задача, реализация, форматы выходов, таблица проверок |
| `.gitignore` | из копии llama.cpp отслеживаются **только** наши файлы, всё остальное игнорируется |

**Что в коммит репозитория (корень), а что в игноре:**

- отслеживается: `AGENTS.md`, `report.md`, `.gitignore`,
  `llama.cpp-hiddenstates/examples/hidden-states/*` (3 файла),
  `llama.cpp-hiddenstates/examples/CMakeLists.txt` (одна строка `add_subdirectory`);
- игнорируется: всё остальное внутри `llama.cpp-hiddenstates` — `src\`, `common\`,
  `vendor\`, `build\`, `.git`-файлы и пр.

Проверка: `git status --short` (в корне) — пусто; `git ls-files` показывает ~7 путей.

У копии `llama.cpp-hiddenstates\.git` **удалён**: родительский репозиторий не может
отслеживать файлы, лежащие внутри вложенного репозитория. История апстрима и та же
сборка доступны в чистом чекауте `D:\llama.cpp` (`git -C D:\llama.cpp status --short`
пустой); при желании можно вернуть `cp -r D:\llama.cpp\.git` в копию.

Собранный бинарник: `llama.cpp-hiddenstates\build\bin\Release\llama-hidden-states.exe`.
Тестовые данные и артефакты: `C:\Windows\TEMP\opencode\hsbatch\`.

## Сборка

Каталог `build\` уже сконфигурирован (переконфигурация нужна только после правок
`CMakeLists.txt`). Сборка многопоточная:

```powershell
$LL = "D:\VScode_pr\confidenceResearch\llama.cpp-hiddenstates"
cmake -S $LL -B $LL\build -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=ON
cmake --build $LL\build --config Release --target llama-hidden-states -- /m
```

Результат: `$LL\build\bin\Release\llama-hidden-states.exe`.

Генератор VS 2022, конфиг Release, бэкенд CUDA (`GGML_CUDA=ON`).
Окружение тестов: Windows / PowerShell, GPU RTX 3050 (6 ГБ),
модель `D:\Lm studio\.lmstudio\models\lmstudio-community\gemma-4-E2B-it-GGUF\gemma-4-E2B-it-Q4_K_M.gguf`
(arch `gemma4`, 35 слоёв, D=1536, SWA 512).

## Запуск

Одиночный режим (v1, поведение заморожено — не менять):

```
llama-hidden-states.exe -m <gguf> -p "<вопрос>" [--answer "<ответ>"] \
    [--out <dir|foo.npy>] [--point-a N] [--dump-tensors] [-b N -ub N -c N -ngl N]
```

Пакетный режим (v2):

```
llama-hidden-states.exe -m <gguf> --batch input.jsonl --out <dir> \
    [--generate] [--max-new-tokens N] [--max-len N] [--no-template] \
    [--no-thinking | --thinking] [--layers all|4,8,12|step=2]
```

Вход пакетного режима — JSONL: `{"id","question","answer"?}` плюс **любые
неизвестные ключи, они копируются в meta.jsonl без изменений**.

Выход: `hidden_a.npy`, `hidden_c.npy` (float16, `[N, L_sel, D]` в v2 / `[L, D]` в v1),
`meta.jsonl`, `config.json`.

Просмотр: `python llama.cpp-hiddenstates\examples\hidden-states\load_example.py <out_dir> --k 3`.

## Пайплайн пробы для новой модели

Протокол пробы **заморожен**: признаки — вектор скрытого состояния слоя L в точке A или C,
стандартизация по train, L2-логистическая регрессия; λ из сетки `{100,300,1000,3000}`
по среднему val AUROC, пара (точка, слой) по max val AUROC, тест считается **один раз**.
При смене модели меняется только `<tag>` и путь к GGUF. Данные те же:
`pipeline/data/main.jsonl` (6000) и `pipeline/data/pilot.jsonl` (200), сплиты уже внутри.
Тег: `main_qwen08`, `main_gemma4e2b`, дальше `main_<модель>_<размер>`.

**0. Подготовка.** Python — `pipeline\.venv\Scripts\python.exe` (numpy; sklearn не нужен).
Все команды — из каталога `pipeline\`, выходы — в `out\<tag>\`.

**1. Пилот 200** (проверить время, обрезки, формы; ~40 с):

```powershell
..\llama.cpp-hiddenstates\build\bin\Release\llama-hidden-states.exe -m <gguf> `
  --batch data\pilot.jsonl --generate --max-new-tokens 64 --max-len 1024 --layers all -ngl 99 `
  --out out\pilot_<tag>
```

Смотреть: `ok=200 skipped=0`, `truncated`, `in ... s (X ex/s)`, `retokenize mismatch`,
`shape=[200, n_layers, n_embd]`. Полный прогон ≈ время пилота × 30.

**2. Полный прогон 6000** (~20 мин на RTX 3050; лог — в файл, т.к. консоль теряет вывод):

```powershell
..\llama.cpp-hiddenstates\build\bin\Release\llama-hidden-states.exe -m <gguf> `
  --batch data\main.jsonl --generate --max-new-tokens 64 --max-len 1024 --layers all -ngl 99 `
  --out out\main_<tag> 2>&1 | Tee-Object out\main_<tag>.log
```

Проверить в конце: `done: 6000/6000 ok=6000 skipped=0`, число `truncated`,
`retokenize mismatch` (~0.5 % — норма), `config.json` (`n_layers`, `n_embd`, `template: true`,
`thinking: false`). Флаги `-b/-ub` не задавать: пакетный режим сам ставит их в `--max-len`.

**3. Разметка** (`label.py`):

```powershell
..\.venv\Scripts\python.exe label.py --meta out\main_<tag>\meta.jsonl --out out\main_<tag>\labels.csv
```

Свериться: `labeled` = 6000 − `truncated` − пустые/вырожденные; доли strict / relaxed /
filtered_a / filtered_b; сколько меток меняет каждый фильтр. Пустые генерации искать по
`logprobs == []`, вырожденные — глазами по `answer` (у Gemma были `}`, `?`, `\r`, `\xa0`,
`<|channel>thought` при `thinking: false`). Все аномалии — в документацию, строки не
удалять (они и так получают пустую метку).

**4. Baseline** (logprob/энтропия первого токена):

```powershell
..\.venv\Scripts\python.exe baseline.py --meta out\main_<tag>\meta.jsonl `
  --labels out\main_<tag>\labels.csv --out out\main_<tag>\baseline.json `
  --scores-out out\main_<tag>\baseline_scores.csv
```

Смотреть test AUROC/AUPRC и ECE сырой `exp(lp0)`: если первый токен почти всегда
служебный, сырая вероятность бесполезна (у Gemma ECE 0.54), это нормально — baseline
всё равно ранжирует.

**5. Проба** (~5–8 мин, CPU, mmap):

```powershell
..\.venv\Scripts\python.exe probe.py --hidden-a out\main_<tag>\hidden_a.npy `
  --hidden-c out\main_<tag>\hidden_c.npy --meta out\main_<tag>\meta.jsonl `
  --labels out\main_<tag>\labels.csv --out out\main_<tag>\probe.json `
  --curve-out out\main_<tag>\probe_curve.csv --scores-out out\main_<tag>\probe_scores.csv
```

Контроли в `probe.json`: перемешанные метки ≈ 0.5 ± 0.06, слой 0 заметно ниже победителя,
разрыв val→test не больше ~0.04 (у Gemma тест оказался выше val — это аномалия, её
описывать, а не «исправлять»). Тест не пересчитывать и слой не перевыбирать.
Детерминизм: повторный запуск с `--out ...\probe2.json --scores-out ...\probe2.csv`
должен дать побайтово те же `probe_curve.csv`/`probe_scores.csv`.
Вместе с `probe.json` пишется `probe_model.npz` (`w`, `mu`, `sd`, `meta`) — переносимая
проба: `score = w[0] + w[1:] @ ((x - mu) / sd)`. Для старых прогонов — `save_probe_model.py
--dir out\main_<tag>` (пересборка детерминированна, тест не пересчитывается).

**6. Сравнение с предыдущей моделью** (метки у моделей разные — это ожидаемо):

```powershell
..\.venv\Scripts\python.exe compare_probes.py --a out\main_qwen08\probe_scores.csv `
  --b out\main_<tag>\probe_scores.csv --split test --out out\compare_qwen08_<tag>_test.json
```

Интерпретация: bootstrap ресэмплит вопросы, у каждой модели своя метка, поэтому ΔAUROC
отражает и разную долю правильных ответов; в тексте это оговаривать.

**7. Документация и коммит.** `report.md` — новый раздел (§N): прогон, аномалии, разметка
со сплитами, baseline, проба (λ-сетка, кривая по слоям, val/test, контроли), сравнение.
`README.md` — новый §3.N по образцу §3.9 + строки в §3.2 (прогоны), §4 (артефакты),
§5 (команды), §6 (ограничения), «Расхождения и недостающие данные». В git — всё, кроме
`*.npy` и логов (игнорируются `.gitignore`).

**Ориентиры (не менять протокол под них):**

| модель | доля правильных (filtered_b) | baseline test AUROC | проба: точка/слой | test AUROC | test AUPRC |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 0.168 | 0.739 (logprob) | C / 13 | 0.857 | 0.593 |
| Gemma 4 E2B | 0.182 | 0.705 (logprob) | C / 19 | 0.935 | 0.744 |

Победивший слой у обеих — ~54 % глубины; если у новой модели он окажется на слое 0 или
качество упадёт до baseline, это результат, а не повод менять сетку λ или точки.

## Модель gpt-oss 20B (отключение reasoning, скорость)

`gpt-oss-20b-MXFP4.gguf` (12.1 ГБ, arch `gpt-oss` = `LLM_ARCH_OPENAI_MOE`, 24 слоя, D=2880,
`l_out` поддержан). Проверено 2026-10-21 на RTX 3050 6 ГБ / 16 ГБ ОЗУ.

**Штатные флаги reasoning не работают:**
- `--no-thinking` шаблон gpt-oss игнорирует — в промпте всё равно `Reasoning: medium`;
- `--reasoning on|off` и `--reasoning-budget` действуют через сэмплер, а генерация в
  инструменте — сырой greedy-argmax без сэмплера (`hidden-states.cpp:846-884`);
- `--chat-template-kwargs` инструмент не передаёт (`common_chat_templates_apply` без kwargs).

**Рабочий обход без правок кода — предзаполнить канал `final`:** в новом JSONL заменить
`question` на готовый harmony-промпт (все остальные ключи строки сохранить), запускать
с `--no-template`. Промпт (одна строка, без переводов внутри):

```
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: <дата>

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|><вопрос>
Answer with just the answer, no explanation.<|end|><|start|>assistant<|channel|>final<|message|>
```

```powershell
llama-hidden-states.exe -m gpt-oss-20b-MXFP4.gguf --batch <final.jsonl> `
  --generate --max-new-tokens 64 --max-len 1024 --layers all --no-thinking --no-template `
  -ngl 11 --out out\main_gptoss20b
```

Модель продолжает сразу в `final`: рассуждений нет, ответы 1–27 токенов, разметка
работает. Проверка на 20 вопросах пилота: 20/20 ok, 0 `truncated`, strict 10/20;
варианты `Reasoning: medium|low|без строки` дают одинаковый текст. Точка A при этом —
последний токен нашего промпта (`<|message|>` канала final), а не `<|start|>assistant`;
это осознанное отклонение, иначе 64 токена уходят в `analysis` и всё обрезается
(0/8 при 64, 4/8 при 128).

**Скорость и память (20 строк, `--max-new-tokens 64`):**

| `-ngl` | с/строка | пик VRAM | примечание |
|---|---|---|---|
| 9 | 2.95 | ~5.5 ГБ | |
| 10 | 2.79 | 5.75 ГБ | |
| **11** | **2.71** | **5.79 ГБ** | оптимум |
| 12 | 6.5–7.8 | 5.79 ГБ | вытеснение в shared memory, в 2.3–2.9 раза медленнее |
| 13 | 5.8–6.3 | 5.75 ГБ | то же |

Вес слоя — 431.8 МиБ (ffn 405, attn 27), поэтому 11 слоёв ≈ 4.6 ГБ + буферы.
`--max-len 256/512`, `-t 12`, `-tb 16`, `-fa on` скорость не меняют; два процесса
параллельно (по `-ngl 5`) дают 40 строк за 277 с вместо 108 с — хуже. GPU занят на
81 %, 46 Вт — не по мощности, а по эффективности ядер MXFP4. Полный прогон 6000:
**~4.5–5 ч** (0.36–0.37 ex/s). Дальнейшее ускорение — только правки кода (батчирование
строк в один ubatch, отказ от второго прохода состояний), не флаги.

### Батчирование строк (`--batch-rows`): статус 2026-10-22 (закоммичено, 3a9ac41)

**Сделано.** Флаг `--batch-rows N` (по умолчанию 1 — прежнее поведение): K строк идут
одним `llama_decode` (`seq_id` на строку, позиции локальные), состояния A/C читаются из
`l_out` по маппингу `row = seq_off[s] + pos`. Последовательности падятся EOS до равной
длины `L_pad`, поэтому `l_out rows == total` (один ubatch). Батчится и генерация:
промпты группы идут префиллом кусками ≤ `max_len` токенов, шаги — по одному токену на
активную строку одним decode; `logprobs`/`entropies`/`truncated`/`n_retok` ведут себя
как раньше. `n_batch = n_ubatch = max_len` сохранён, группа состояний режется так,
чтобы `K*L_pad ≤ max_len`.

**Раскладка подтверждена — seq-major.** Дамп слоя 0 при K=4: строки, равные row0, —
это `[0, 108]`; A/C из захвата совпали со строками 105/213/321/429 (diff 0.01 = fp16).
Гипотеза pos-major опровергнута.

**Батч ≠ одиночный проход численно (раскладка верна).** Слой 0: median absdiff 3.8e-6
(округление fp16), растёт с глубиной, усиливается MoE-роутингом; относительно нормы
0.1–0.5 %, cosine median 1.00000, min 0.9987 (200 строк, A b8 vs эталон). Утечки между
строками нет: состояние seq0 не зависит от соседа (diff 0.0). Не влияют `--swa-full`,
`kv_unified`, CPU/GPU. Детерминизм: два прогона b8 → `hidden_a.npy`, `hidden_c.npy`,
`meta.jsonl`, `config.json` побайтово равны.

**Снят ложный симптом «batch K=1 vs v1 = 652»**: разная токенизация промпта (argv
многострочный → 106 токенов, тот же в одну строку → 105). При одинаковых токенах
(`nl.jsonl`, 107) слои 0–22 совпадают точно, слой 23 — 0.25 (C 0.06).

**Замеры, 200 строк** (`gptoss20b_final200_ans.jsonl`, только состояния;
`gptoss20b_final200_q.jsonl`, с генерацией):

| режим | `--batch-rows` | время, с | ex/s | к b1 |
|---|---|---|---|---|
| только состояния | 1 | 229.5 / 268.3* | 0.87 | — |
| только состояния | 4 | 68.6 / 96.9* | 2.91 | ×3.3 |
| только состояния | 8 | 43.7 / 76.7* | 4.58 | ×5.2 |
| с генерацией | 1 | 533 | 0.37 | — |
| с генерацией | 4 | 235.2 | 0.85 | ×2.3 |
| с генерацией | 8 | 193.2 | 1.04 | ×2.8 |

\* второй сеанс, GPU под нагрузкой/троттлингом. State-only b8 после правок — 44.9 с
(4.46 ex/s), массивы побайтово совпадают с прежней сборкой.

**Ответы и метки.** b1 побайтово совпадает с прежним одиночным эталоном (0 расхождений
на 20). b8: 170/200 ответов совпадают текстово, 30 отличаются формулировкой
(`Rudyard Kipling` с тонким пробелом, кавычки, «specifically» и т.п.), но метки по
замороженным определениям — **0 расхождений** на 200 (strict / relaxed / filtered_a /
filtered_b / correct). 20 строк: 0 skipped, 0 truncated; смесь строк с ответами и без —
10/10; при `--max-len 64` все строки skipped с прежней причиной, код возврата 0.

Артефакты: `C:\Windows\TEMP\opencode\hsbatch\` (`gen200q_b8_batched5` — эталон b8 с
генерацией, `gen200q_b4_batched`, `gen20_b8_v6/v7` — детерминизм, `gen20_b1_v2` —
регрессия b1, `ans200_b8_v2`, `mix10_b8`, `skip_len64`, `odql_dup8_b8`; старые
`gen200q_b8`, `ans200_b1/b4/b8`, `final20`, `dup2*`, `v1_p*`). Входы:
`gptoss20b_final20.jsonl`, `gptoss20b_final20_ans.jsonl`, `gptoss20b_final200_ans.jsonl`,
`gptoss20b_final200_q.jsonl`, `mix_ans_q10.jsonl`, `odql_dup8.jsonl`.

### Полный прогон 6000 с батчингом (2026-10-22)

Вход `gptoss20b_final6000_q_v2.jsonl` (собран `pipeline/make_gptoss_prompt.py`,
инструкция ровно один раз), флаги `--generate --max-new-tokens 64 --max-len 1024
--layers all --no-thinking --no-template -ngl 11 --batch-rows 8`.

| | значение |
|---|---|
| итог | **6000/6000 ok, 0 skipped, `truncated` 27 (0.45 %)** |
| время | **10733.5 s, 0.56 ex/s** (первые ~2100 строк — 1.06 ex/s) |
| `retokenize mismatch` | 21/6000 (0.3 %) |
| массивы | `[6000, 24, 2880]` f16, по 829.4 МБ |
| разметка | strict 0.494, filtered_b 0.490, null 27 (все `truncated`) |
| baseline test | logprob 0.828, entropy 0.841 |
| проба | λ = 3000, **C / 16**, val 0.9157, test **0.9248** [0.909, 0.939], AUPRC 0.9197 |
| контроли | перемешанные метки 0.494 ± 0.074, слой 0 C 0.684 |

Падение темпа во второй половине — нехватка RAM (свободно 0.7–1 ГБ из 16 при 12 ГБ
процесса), GPU 99 %, 53 °C, без троттлинга; тот же набор до исправления входа прошёл за
5596 s (1.07 ex/s). Каталоги: `pipeline\out\main_gptoss20b_v2` — действительный прогон;
`main_gptoss20b` (325/6000, остановлен) и `main_gptoss20b_dupinstr` (дефектный промпт) —
не используются. Пилоты малых моделей: `hsbatch\qwen_p20_*`, `gemma_p20_*` (b8 корректен,
но выигрыш ~×1.2, у Gemma 3/20 ответов иначе при 0 расхождений меток).

## Модели с неполным `l_out` (nemotron_h) и проверка прогона

`NVIDIA-Nemotron-3-Nano-4B` (arch `nemotron_h`, 42 слоя, D=3136) — гибрид: `l_out`
эмитится только в `build_ffn_layer` (`src/models/nemotron-h.cpp:255`), поэтому он есть
лишь на 17 слоях: `1,3,5,8,10,13,15,18,20,22,25,27,29,33,37,39,41`. SSM/attention слои
отдают только `nemotron_h_block_out`.

**Инцидент.** Старый код требовал захвата **всех** выбранных слоёв
(`st.n_cap_c != L_sel` → skipped). `--layers all` выбирал 42 слоя, захватывалось 17,
и каждая строка писалась как `skipped`, а в массивах оставались нули. Так потеряны
`out\main_nemotron3nano4b` (5269 строк, 2.7 ч) и `out\pilot_nemotron3nano4b`
(там же перезапуск в тот же каталог перезаписал `config.json`); оба помечены
`README-invalid.md`, в анализ не идут, но не удаляются.

**Что исправлено (2026-10-27):**

- Перед прогоном инструмент делает один пробный decode на BOS, собирает реально
  существующие `l_out-N` и выбрасывает выбранные слои без `l_out` (предупреждение в
  лог, исходный список — в `config.json` как `layers_requested`). Если `l_out` есть на
  всех слоях, поведение не меняется.
- `write_skipped` больше не затирает сгенерированный ответ: `generated`, `truncated`,
  `n_tokens`, `pos_a`, `pos_c`, `logprobs`, `entropies` пишутся как есть.
- `probe.py` индексирует массивы по `row` из meta (а не по позиции в файле), сверяет
  формы, `n_embd` и список слоёв с `config.json`, пропускает all-zero слои и падает с
  внятной ошибкой, если в каком-то сплите нет строк.
- `baseline.py`/`probe.py` сообщают номер битой строки meta вместо трейсбека; sigmoid
  не переполняется на больших скорах.
- Новый `pipeline/verify_run.py` — единая проверка каталога прогона (см. таблицу
  проверок). Запускать **до** `probe.py`.

**Первый валидный прогон** (`out\pilot_nemotron3nano4b_v2`, 200 строк,
`--layers all -ngl 99`, batch-rows 1): 200/200 ok, 0 skipped, 0 truncated, 44.7 с
(4.48 ex/s), массив `[200, 17, 3136]`, `verify_run.py --full` проходит. Разметка:
strict 64/200 = 0.320, filtered_b 62/200 = 0.310; baseline test — logprob 0.801,
entropy 0.831. Пилотная проба (120/40/40): λ=100, C / слой 13 (индекс 5), val 0.851,
test 0.701 (7 положительных, интервал широкий) — это проверка конвейера, не результат.

Батчинг здесь не помогает: 50 строк, `--batch-rows 1` — 10.4 с, `--batch-rows 4` —
10.4 с (доминирует генерация). Ориентир полного прогона 6000: 25–60 мин.

### Полный прогон 6000 (2026-10-28)

`out\main_nemotron3nano4b_v2`, флаги `--generate --max-new-tokens 64 --max-len 1024
--layers all -ngl 99` (batch-rows 1):

| | значение |
|---|---|
| итог | **6000/6000 ok, 0 skipped, `truncated` 5 (0.08 %)** |
| время | **1320.6 s (22 мин), 4.54 ex/s** |
| `retokenize mismatch` | 19/6000 (0.3 %) |
| массивы | `[6000, 17, 3136]` f16, по 610 МБ |
| `layers_requested` | 42 → `layers` 17 |
| разметка | strict 0.295, filtered_b 0.285, null 5 (все `truncated`) |
| baseline test | logprob 0.802, entropy 0.816 |
| проба | λ = 3000, **C / слой 20** (49 %), val 0.9059, test **0.8819** [0.860, 0.901], AUPRC 0.7357 |
| контроли | перемешанные метки 0.481 ± 0.035, слой 0 C 0.650 |

Место среди моделей: по времени Nemotron — самый медленный из трёх прогонов малых
моделей, но разрыв мал: 1320.6 s против 1258.1 s (Gemma) и 1287.1 s (Qwen), 3–5 %;
gpt-oss — 10733.5 s (3 ч).
По качеству проб Nemotron между малой парой и 20B: лучше Qwen (+0.026 AUROC, интервал
задевает ноль), хуже Gemma (−0.054) и gpt-oss (−0.043). Полные таблицы — `report.md` §17,
`README.md` §3.12. Кросс-скоры: `out\cross_*_nemotron_{test,val}.json`.

## Сохранность данных (не перезаписывать артефакты)

Собранные данные дорогие и не воспроизводятся «на лету» (прогон 6000 — ~20 минут на
модель). **Ничего не перезаписывать и не удалять без явной просьбы пользователя.**

- Каждый прогон — в **свой** каталог `out\main_<tag>\` / `out\pilot_<tag>\`; тег не
  повторять. Не запускать инструмент с `--out` поверх существующего каталога: пакетный
  режим делает resume по `meta.jsonl` и допишет/перезапишет строки.
- Существующие `meta.jsonl`, `labels.csv`, `baseline.*`, `probe.*`, `hidden_*.npy`
  считать окончательными: новые вычисления — в новые файлы (`probe2.*`, новый тег),
  оригиналы только сравнивать.
- `.npy` большие (сотни МБ) и не входят в git — они живут только на диске. Не чистить
  `pipeline\out\` и `C:\Windows\TEMP\opencode\`, не удалять пилоты и промежуточные
  каталоги «для экономии места».
- После любого прогона сверять: строк в `meta.jsonl` == ожидаемому N, `ok + skipped`
  совпадает, формы массивов == `[N, n_layers, D]`, `probe_scores.csv` == val + test.
- Если файл всё же нужно пересчитать (например, проверка детерминизма), писать в
  отдельные имена и убедиться, что оригинал не изменился (`git status`, размер,
  SHA-256).

## Инварианты (нарушать нельзя)

1. **Определения точек** (зафиксированы):
   - **A** = последний токен промпта, уже отрендеренного чат-шаблоном модели с
     `add_generation_prompt=true`; состояние, из которого выдаётся первый токен ответа.
   - **C** = последний содержательный токен ответа, **без** EOG/конца реплики.
   - Промпт и ответ токенизируются **раздельно** (`parse_special=true`), затем id склеиваются.
2. **Состояния считаются отдельным проходом** по «промпт + ответ» (A и C из одного
   прохода). Генерация (`--generate`) идёт другим проходом; перед проходом состояний
   обязателен `llama_memory_clear(mem, true)`.
3. **`n_batch = n_ubatch = --max-len`** и `n_tokens <= max_len`. Это гарантирует один
   ubatch, т.е. индекс строки тензора == индекс токена. Слишком длинные примеры
   **пропускаются** с причиной в meta, прогон не падает.
4. **`logits=true` минимум на одной позиции** — иначе срабатывает
   `GGML_ASSERT(n_outputs >= 1)` в `graph_reserve` (`src/llama-context.cpp:1748-1750`).
5. **`params.warmup = false`** — иначе warmup-decode вызовет колбэк на 1–2 токенах.
6. **Колбэк `cb_eval`**: в фазе `ask` возвращать `true` **только** для `l_out-*`
   (возврат `false` для остальных не даёт синхронизаций); в фазе `ask=false` всегда
   возвращать `true`, иначе `break` оборвёт вычисление графа.
   Тензоры `l_out-0 … l_out-34` создаются в рантайме (`src/llama-context.cpp:2454-2460`),
   источник `cb(cur, "l_out", il)` — `src/models/gemma4.cpp`; строки `l_out-N` полные
   (`ne[1] == n_tokens`), `out_ids` влияет только на `result_norm`/`result_output`.
7. **Свои флаги вынимаются из `argv` до `common_params_parse`** — общий парсер
   отвергает неизвестные аргументы.

## Структура кода (`hidden-states.cpp`)

- `hs_state` + `cb_eval` — состояние колбэка; буферы `hs_c`/`hs_a` (`L_sel * D`)
  выделяются один раз и переиспользуются; `enabled=false` отключает захват
  (генерация идёт без синхронизаций на слоях).
- `parse_layers("all" | "4,8,12" | "step=2")` → `layer_row[layer]` (−1 = не выбран).
- `npy_write_header` / `write_npy_f16` (v1, 2-D, последовательная запись) /
  `npy_f16_writer` (v2, 3-D, случайный доступ). **Смещение данных = размер заголовка**;
  файл дотягивается до полного размера, чтобы пропущенные строки читались как нули.
- `decode_ids()` — единственный способ декодировать (кладёт `logits` по нужным позициям).
- `run_batch()` — цикл по строкам: шаблон → токенизация → генерация (если нужно) →
  срез EOG → `llama_memory_clear` → проход состояний → запись строки npy + meta.
  Возобновление по `id` из `meta.jsonl`, flush каждые 16 примеров.
- `main()` — парсинг своих флагов, затем либо `run_batch`, либо v1-ветка.

## Проверки (повторять после любых правок)

| Проверка | Ожидание |
|---|---|
| Пакетный vs одиночный (`--no-template`, `-b 1024 -ub 1024 -c 1024` vs `--max-len 1024`) | `hidden_*.npy[0]` байт-в-байт == `[L, D]` из v1 |
| Детерминизм: два одинаковых прогона | SHA-256 `.npy` совпадают |
| Инвариантность A: запись с `answer` и без | векторы A байт-в-байт идентичны; без ответа A == C |
| NaN/Inf | нет |
| `--layers step=2` / `4,8,12` | строки == соответствующим слоям полного прогона |
| Слишком длинный | `status: skipped` + причина, код возврата 0 |
| Возобновление | обрабатываются только id, которых нет в meta.jsonl; массивы восстановимы |
| TriviaQA-20, `--generate --max-new-tokens 64` | 20/20 ok; доля `truncated` ~20 % |
| `python verify_run.py --dir out/main_<tag>` (перед `probe.py`) | 0 failed: формы `[N, L_sel, D]`, `row` без дыр, ok-строки непустые, ни одного нулевого слоя, метки не пустые, есть все три сплита |

Полные числа и таблицы — в `report.md`, раздел 7.

## Перед долгими прогонами (цена ошибки — часы)

Вход и флаги проверять **до** запуска: прогон 6000 строк — это 20 мин (малые модели)
или 1.5–4.5 ч, а дефект подготовки входа виден только по результату.

- Сгенерировать вход и **посмотреть глазами первые 2–3 строки** `question` (начало,
  конец, число вхождений инструкции: `count("Answer with just the answer") == 1`).
- Сравнить промпт с эталоном, на котором снят пилот (`gptoss20b_final20*.jsonl`,
  `pilot.jsonl`-вход): тот же шаблон, та же инструкция, то же число вхождений.
- Проверить, что сборщик **не дублирует** то, что уже есть в `question`: после
  `prepare_data.py` вопрос оканчивается на `\nAnswer with just the answer, no explanation.`.
- Проверить идемпотентность: повторный запуск скрипта даёт побайтово тот же файл.
- Сверить число строк и отсутствие пустых `question`. Дубль `id` `bb_956` в
  `main.jsonl` — норма (две разные формулировки вопроса), он должен остаться.
- Прогнать пилот 20–50 строк на **том же** входе (30–60 с) и только потом полный набор.

**Инцидент 2026-10-22.** Первый полный прогон gpt-oss (6000 строк, 5596 с) сделан на
входе, где инструкция «Answer with just the answer, no explanation.» стояла **дважды**:
она уже есть в `question` из `main.jsonl`, а сборщик harmony-промпта добавил её ещё раз.
Нашлось только после прогона; результат сохранён как `main_gptoss20b_dupinstr` и в отчёты
не пошёл, прогон пришлось повторять. Отсюда правило: скрипты подготовки входа проверять
до запуска, а не по результату многочасового прогона.

## Коммиты

- В сообщения коммитов **не** добавлять строку `Assisted-by:` и любые другие
  упоминания AI-ассистентов (`Co-authored-by`, `Generated-by` и т.п.). Коммиты
  оформляются только от имени автора (`Fatalist0001`).

## Чего не делать

- Не менять v1-ветку и её формат выхода (`[L, D]` + `hidden_states.json`).
- Не править ядро llama.cpp — инструмент написан только на публичном API
  (`llama.h`, `common/*.h`, включая `chat.h`, `build-info.h`).
- Не добавлять многопоточность и лишние флаги (договорённость: «не усложнять»).
- Не удалять и не перезаписывать собранные артефакты (`out\*`, `.npy`, `meta.jsonl`,
  `labels.csv`, `probe.*`, `baseline.*`) без явной просьбы пользователя.
- Не запускать `--out` в существующий каталог прогона.
- Не менять фиксированные определения точек A и C.

## Текущее состояние (2026-10-27)

- **Nemotron 3 Nano 4B: полный прогон ещё не сделан.** Инструмент исправлен
  (автоопределение `l_out`, 17 слоёв), конвейер проверен на пилоте
  `pipeline\out\pilot_nemotron3nano4b_v2` (200/200 ok, 4.48 ex/s) и на 50 строках со
  всеми сплитами. Старые `main_nemotron3nano4b` / `pilot_nemotron3nano4b` непригодны
  (`README-invalid.md`); поддельные `*_expected.*`, `probe_simple.json`, `baseline.json`
  и скрипты-обходы удалены. Следующий шаг — `main_nemotron3nano4b_v2` (6000 строк,
  `--layers all -ngl 99`, batch-rows 1), затем `label.py` → `verify_run.py` →
  `baseline.py` → `probe.py`.
- Коммиты по gpt-oss и батчингу (включая `3a9ac41`, `f33f450`, `7ea246c`, `b76dae1`) —
  в `origin/main`. Коммит `3ad2230` добавил фиктивные результаты Nemotron (0.500 AUROC
  на нулевых массивах) — артефакты удалены, исправление пока не закоммичено.
- **gpt-oss-20b прогнан полностью** (6000 строк, `--batch-rows 8 -ngl 11`, вход
  `gptoss20b_final6000_q_v2.jsonl`): 6000/6000 ok, truncated 27, 10733.5 s (0.56 ex/s),
  метки strict 0.494 / filtered_b 0.490, baseline test 0.828/0.841, проба C/16
  test 0.9248. Каталог `pipeline\out\main_gptoss20b_v2` — действительный прогон.
- `pipeline\out\main_gptoss20b` (325/6000, остановлен вручную) и
  `main_gptoss20b_dupinstr` (промпт с дублем инструкции) — не используются, в git не
  входят, но и не удаляются.
- Документация: §15–§16 в `report.md`, §3.10/§3.11 и правки §3.2/§4/§5/§6 в `README.md`,
  раздел про полный прогон в `AGENTS.md`.
- Веса выбранных проб сохраняются в `pipeline\out\main_<tag>\probe_model.npz`
  (`w`, `mu`, `sd`, JSON `meta`); для старых прогонов — `save_probe_model.py --dir`.
- Кросс-модельные скоры (проба одной модели → метка другой, `cross_probe_scores.py`,
  6 JSON в `pipeline\out\cross_*_{test,val}.json`), тест: gpt-oss→Gemma 0.832,
  gpt-oss→Qwen 0.801, Qwen→Gemma 0.771, Gemma→Qwen 0.753, Gemma→gpt-oss 0.690,
  Qwen→gpt-oss 0.679 при родных 0.86–0.94. Это сравнение предсказаний, а не перенос
  весов: веса между D=1024/1536/2880 не переносятся.
- Стратификация по метке источника (`cross_probe_strata.py`, 12 JSON
  `pipeline\out\cross_strata_*`): внутри страт 0.59–0.82, добавка скора к «метке A»
  в логистике +0.05…+0.10 — сигнал сверх бинарного исхода, но контроль на длину
  вопроса/logprob не делался.
- Пилоты малых моделей с батчингом (`hsbatch\qwen_p20_*`, `gemma_p20_*`): работает,
  но выигрыш ~×1.2; для массовых прогонов b8 оправдан только на gpt-oss.
