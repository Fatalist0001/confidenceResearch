# Отчёт: извлечение скрытых состояний из GGUF-модели через llama.cpp

**Дата:** 2026-10-12
**Окружение:** Windows, llama.cpp собран с CUDA (GPU: RTX 3050, 6 ГБ)
**Модель для проверки:** `gemma-4-E2B-it-Q4_K_M.gguf` (arch `gemma4`, 35 слоёв, D=1536)

---

## 1. Задача

Минимальный CLI-инструмент поверх llama.cpp, который извлекает скрытые состояния **всех слоёв** модели для двух интересующих позиций последовательности.

Два режима:

- **v1, одиночный** — один пример через `-p` / `--answer` (исходный режим, не изменён);
- **v2, пакетный** — `--batch input.jsonl` для массовых прогонов (см. раздел 6).

Точки (определения в разделе 6.2): **A** — последний токен промпта, отрендеренного чат-шаблоном с `add_generation_prompt`; **C** — последний содержательный токен ответа без EOG.

Выход: `.npy` float16 (`[L, D]` в v1, `[N, L_sel, D]` в v2) + json/jsonl с метаданными.

---

## 2. Что сделано

### 2.1. Исследование модели и механизмов llama.cpp

Прочитаны и подтверждены:

- Вывод всех имён тензоров графа (`examples/eval-callback`) → доказано, что тензоры `l_out-0 … l_out-34` существуют для **всех 35 слоёв**;
- семантика `cb_eval` в `ggml/src/ggml-backend.cpp:1677-1710` (фазы `ask=true/false`);
- именование `l_out-N` создаётся в рантайме через `llm_graph_cb` (`src/llama-context.cpp:2454-2460`);
- источник `cb(cur, "l_out", il)` в архитектуре модели — `src/models/gemma4.cpp`;
- `out_ids` влияет только на `result_norm`/`result_output`, строки `l_out-N` остаются полными;
- ассерты размеров батча в `src/llama-context.cpp:1748-1750`.

### 2.2. Новый пример (таргет `llama-hidden-states`)

Инструмент лежит в каталоге `llama.cpp-hiddenstates\` — это полная копия llama.cpp (коммит
`d4cff114c`) с нашими двумя изменениями; оригинальный чекаут `D:\llama.cpp` возвращён
в исходное состояние и не используется.

| Файл (внутри `llama.cpp-hiddenstates\`) | Описание |
|---|---|
| [`examples/hidden-states/hidden-states.cpp`](file:///D:/VScode_pr/confidenceResearch/llama.cpp-hiddenstates/examples/hidden-states/hidden-states.cpp) | основной код инструмента (одиночный + пакетный режимы) |
| [`examples/hidden-states/CMakeLists.txt`](file:///D:/VScode_pr/confidenceResearch/llama.cpp-hiddenstates/examples/hidden-states/CMakeLists.txt) | CMake-таргет `llama-hidden-states` |
| [`examples/hidden-states/load_example.py`](file:///D:/VScode_pr/confidenceResearch/llama.cpp-hiddenstates/examples/hidden-states/load_example.py) | просмотр результатов пакетного режима (npy через `mmap_mode="r"` + meta.jsonl) |
| [`examples/CMakeLists.txt`](file:///D:/VScode_pr/confidenceResearch/llama.cpp-hiddenstates/examples/CMakeLists.txt) | подключён `add_subdirectory(hidden-states)` |

Сборка (каталог `build\` сконфигурирован; сборка многопоточная):

```powershell
$LL = "D:\VScode_pr\confidenceResearch\llama.cpp-hiddenstates"
cmake -S $LL -B $LL\build -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=ON
cmake --build $LL\build --config Release --target llama-hidden-states -- /m
```

Результат: `$LL\build\bin\Release\llama-hidden-states.exe`.

### 2.3. Ключевые решения в коде

1. **Парсинг аргументов.** Свои флаги (`--answer`, `--out`, `--point-a`, `--dump-tensors`) вынимаются из `argv` **до** вызова `common_params_parse`, иначе общий парсер отвергает неизвестные аргументы. Стандартные `-m`, `-p`, `-ngl`, `-b`, `-ub`, `-c` работают как обычно.

2. **Двухфазная инициализация** (гарантирует `n_ubatch >= n_tokens`, т.е. весь батч — один ubatch, и индекс строки тензора == индекс токена):
   - `common_init_from_params(params, model_only=true)` — загружаем только модель, чтобы получить словарь;
   - токенизируем вопрос и ответ **раздельно**, склеиваем id (так индекс A точный);
   - вычисляем `n_tokens`, `pos_a`, `pos_c`, затем `n_batch`/`n_ubatch`/`n_ctx` подгоняем под размер батча;
   - `llama_init_from_model(model, common_context_params_to_llama(params))`.

3. **Колбэк `cb_eval`.** В фазе `ask` возвращаем `true` только для тензоров с префиксом `l_out-` (остальные считать не нужно — это сильно ускоряет проход). В фазе вычисления читаем ровно одну строку `pos_c` (и `pos_a`) через `ggml_backend_tensor_get`, поддержаны типы F32 / F16 / BF16. `params.warmup = false` обязателен, иначе warmup-decode вызовет колбэк на 1–2 токенах и заполнит буфер мусором.

4. **`logits=true` на позициях A и C** — чтобы строки не отбрасывались и чтобы удовлетворить `GGML_ASSERT(n_outputs >= 1)` в `graph_reserve`.

5. **Запись `.npy`** вручную (numpy v1.0, little-endian `<f2`, `fortran_order=False`), конвертация `ggml_fp32_to_fp16`.

6. **Валидация:** проверка на `NaN/Inf`, лог min/max/mean по каждому вектору, проверка что все 35 слоёв фактически захвачены.

---

## 3. Формат выходных файлов

При `--out <dir>`:

| Файл | Содержимое |
|---|---|
| `<dir>/hidden_c.npy` | скрытые состояния всех слоёв в позиции C, `[35, 1536]`, float16 |
| `<dir>/hidden_a.npy` | то же для позиции A |
| `<dir>/hidden_states.json` | модель, размерности, `n_tokens`, `pos_a`, `pos_c`, пути к файлам |

При `--out foo.npy`: `foo.npy` (C), `foo_a.npy` (A), `foo.json`.

Пример `hidden_states.json`:

```json
{
  "model": "Gemma 4 E2B",
  "n_layers": 35,
  "n_embd": 1536,
  "n_tokens": 11,
  "n_prompt_tokens": 9,
  "n_answer_tokens": 2,
  "pos_a": 8,
  "pos_c": 10,
  "files": { "c": "...hidden_c.npy", "a": "...hidden_a.npy" }
}
```

---

## 4. Результаты проверки

Пример запуска:

```powershell
llama-hidden-states.exe -m <gguf> -p "What is 2 + 2?" --answer " 4" --point-a 3 --out <dir>
```

| Проверка | Результат |
|---|---|
| Тензоры `l_out-0 … l_out-34` присутствуют (через `--dump-tensors`) | все 35 слоёв найдены |
| Форма выхода | `[35, 1536]`, dtype `float16` |
| NaN / Inf | отсутствуют |
| **Детерминизм** (два одинаковых запуска) | `.npy` байт-в-байт идентичны (совпадение SHA-256) |
| **Инвариантность A**: вектор A из прохода «вопрос + ответ» vs проход «только вопрос» (`--answer=`) | **байт-в-байт идентичны** — ответ не влияет на состояния до него |
| `pos_a=0` vs `pos_a=8` | заметно различаются (maxabs = 72.2) — состояния не вырождены |
| A vs C на последнем слое | различаются (maxabs = 12.8) |

Тестовые артефакты: `C:\Windows\TEMP\opencode\hs_run{1,2,3}`, `hs_point*`.

---

## 5. Использование: одиночный режим (v1)

```
llama-hidden-states.exe -m <model.gguf> -p "<вопрос>" [--answer "<ответ>"] [--out <dir|file.npy>] [--point-a N] [--dump-tensors] [общие флаги llama.cpp]
```

- `-p` — вопрос (обязателен);
- `--answer` — ответ, который подклеивается к вопросу (может быть пустым: `--answer=`);
- `--point-a N` — 0-based индекс токена внутри вопроса (по умолчанию — последний токен вопроса);
- `--out` — папка или путь вида `foo.npy`;
- `--dump-tensors` — печатает все имена тензоров графа (для отладки);
- остальные флаги (например `-ngl 99`, `-c`, `-ub`) передаются в общий парсер llama.cpp.

Пустой ответ полезен для сверки: `--answer=` даёт позицию C == позиции A, и `hidden_a.npy` такого прогона должен совпадать с `hidden_a.npy` прогона с ответом.

Режим v1 не изменён: те же флаги, тот же формат выхода (`hidden_a.npy` / `hidden_c.npy` формы `[L, D]` + `hidden_states.json`).

---

## 6. Пакетный режим (v2, `--batch`)

Новый режим добавлен **рядом** со старым и включается только флагом `--batch`.

```
llama-hidden-states.exe -m <model.gguf> --batch input.jsonl --out <dir> \
    [--generate] [--max-new-tokens N] [--max-len N] [--no-template] \
    [--no-thinking | --thinking] \
    [--layers all|4,8,12|step=2] [общие флаги llama.cpp]
```

### 6.1. Вход

JSONL, по одному объекту в строке:

```json
{"id": "t1", "question": "What is 2 + 2?", "answer": " 4"}
{"id": "t2", "question": "What is the capital of France?"}
{"id": "t3", "question": "What is the capital of Australia?", "gold": "Canberra", "dataset": "triviaqa-dev"}
```

`answer` необязателен. **Любые неизвестные ключи** (`gold`, `dataset`, …) **копируются в `meta.jsonl` без изменений**.

### 6.2. Определения точек A и C (фиксированные)

| Точка | Определение |
|---|---|
| **A** | последний токен промпта, **уже отрендеренного чат-шаблоном модели** вместе с генерационным префиксом (`add_generation_prompt=true`). Это состояние, из которого модель выдаёт первый токен ответа. |
| **C** | последний **содержательный** токен ответа, **без** токенов конца реплики / EOG. |

Промпт и ответ токенизируются **раздельно** (`parse_special=true`, шаблон уже отрендерен), затем id склеиваются; A = `len(prompt_tokens) - 1`, C = `len(all_tokens) - 1`. Токены EOG в хвосте ответа отрезаются до расчёта C.

Определения зашиты в `config.json` (`point_definitions`).

### 6.3. Как считается

1. Модель и контекст загружаются **один раз** на весь файл. Между примерами — `llama_memory_clear(mem, true)`; буферы колбэка (`hs_c`/`hs_a`, размер `L_sel * D`) переиспользуются и выделяются один раз.
2. Чат-шаблон берётся из самого GGUF (`common_chat_templates_init` + `common_chat_templates_apply`, `add_generation_prompt=true`). Флаг `--no-template` — сырой текст (без шаблона, BOS по `llama_vocab_get_add_bos`).
   По умолчанию в шаблон передаётся `enable_thinking=false` (`--no-thinking`); флаг `--thinking` возвращает старое поведение шаблона. Значение пишется в `config.json` → `thinking`.
3. Если `answer` отсутствует и указан `--generate` — жадная генерация (argmax, temp 0), до `--max-new-tokens` или до EOG. EOG в ответ не добавляется. Если генерация остановилась **не** по EOG (упёрлась в `--max-new-tokens` / `--max-len`), строка **не пропускается**, а помечается `"truncated": true` в `meta.jsonl`.
4. Состояния **всегда** считаются отдельным проходом по «промпт + ответ» (A и C из одного прохода). `n_batch = n_ubatch = --max-len` (по умолчанию 1024), `n_ctx = max_len`. Слишком длинные примеры **пропускаются** с записью причины в `meta.jsonl` — прогон не падает.
5. `--layers` — подмножество слоёв: `all` (по умолчанию), `4,8,12`, `step=2`. Слои хранятся в порядке возрастания; строка i массивов = `layers[i]`.

### 6.4. Выход (`--out dir`)

| Файл | Содержимое |
|---|---|
| `hidden_a.npy` | float16, форма `[N, L_sel, D]` (N = число строк входа), пишется через memmap; заголовок npy создаётся сразу по полному числу строк |
| `hidden_c.npy` | то же для точки C |
| `meta.jsonl` | по строке на пример: `id`, `row`, `n_tokens`, `pos_a`, `pos_c`, `answer`, `generated`, `truncated`, `status` (`ok` / `skipped`) + `reason`; **плюс все неизвестные ключи входной строки, скопированные без изменений** |
| `config.json` | модель, слои, `max_len`, `max_new_tokens`, `generate`, шаблон да/нет, **thinking да/нет**, определения A/C, версия llama.cpp (`llama_build_number`, `llama_commit`, `ggml_version`) |

**Возобновление.** При повторном запуске в тот же каталог уже обработанные `id` из `meta.jsonl` пропускаются. `flush` на диск каждые 16 примеров (npy + meta).

**Прогресс** в консоль: `[done/total] ok=… skipped=… truncated=…  N ex/s`.

### 6.5. Просмотр результата

```powershell
python llama.cpp-hiddenstates\examples\hidden-states\load_example.py <out_dir> --k 3
```

Читает `hidden_*.npy` через `np.load(mmap_mode="r")`, `meta.jsonl` и `config.json`, печатает формы, dtype, признак finiteness и несколько строк метаданных.

---

## 7. Проверки пакетного режима

| Проверка | Результат |
|---|---|
| **Байтовое совпадение** пакетного и одиночного режима (`--no-template`, те же токены, `-b 1024 -ub 1024 -c 1024` vs `--max-len 1024`) | `hidden_c.npy[0]` и `hidden_a.npy[0]` **байт-в-байт идентичны** векторам `[L, D]` из v1 |
| **Детерминизм**: два прогона одного файла | `.npy` совпадают по SHA-256 |
| **Инвариантность A**: запись с `answer` и без `answer` | векторы A **байт-в-байт идентичны**; в прогоне без ответа A == C |
| NaN / Inf | отсутствуют |
| Все слои из `--layers` захвачены | `--layers step=2` → `[5, 18, 1536]`, `--layers 4,8,12` → `[5, 3, 1536]`; строки совпадают с соответствующими слоями полного прогона `[5, 35, 1536]` |
| Слишком длинный пример | `--max-len 5` → `skipped (too long: 11 tokens > max_len 5)`, прогон завершается с кодом 0 |
| Чат-шаблон из GGUF | работает (без `--no-template`); `config.json: template=true` |
| **Возобновление** | из 5 примеров в `meta.jsonl` оставлено 3, строки 3–4 массивов занулены → повторный прогон: `resuming: 3 id(s)`, обработано 2, массивы восстановлены байт-в-байт |
| EOG в генерации | `--generate` на «What is 2 + 2?» → `answer=" 4"`, `n_tokens=25` (23 промпт + 2 ответа), EOG не попал ни в ответ, ни в C |
| **50 вопросов с `--generate`** (`--max-new-tokens 32 --layers step=2`, на момент прогона шаблон рендерился с `enable_thinking=true`) | 50/50 ok, **27.7 s**, **1.81 ex/s**, массивы `[50, 18, 1536]` float16, все значения конечны |
| Регрессия после доработок: пакетный vs одиночный | по-прежнему **байт-в-байт** |
| `--no-thinking` (по умолчанию) vs `--thinking` | промпт короче: 15–18 токенов против 22–25; ответы различаются; `config.json: thinking` = `false` / `true` |
| Пометка обрыва `truncated` | на TriviaQA-20: **4/20 (20 %)** с `--no-thinking`, **19/20** с `--thinking`; такие строки получают `status: ok` и `truncated: true` |
| Сквозные поля | `gold`, `dataset`, `split` из входного JSONL продублированы в `meta.jsonl` без изменений |

### 7.1. TriviaQA-стиль, 20 вопросов, `--generate --max-new-tokens 64`

Вход: `in_trivia20.jsonl` — 20 вопросов, в каждом дополнительно `gold`, `dataset`, `split` (для проверки сквозных полей).

```
llama-hidden-states.exe -m <gguf> --batch in_trivia20.jsonl --generate \
    --max-new-tokens 64 --max-len 1024 --layers step=2 --out out_trivia20
```

Результат: **20/20 ok, 0 skipped, truncated = 4 (20 %)**, 9.4 s, **2.12 ex/s**, массивы `[20, 18, 1536]` float16, конечны. `--no-thinking` (по умолчанию) — `config.json: thinking=false`.

Доля обрывов: **4 / 20 = 20 %**.

5 ответов:

| id | truncated | ответ |
|---|---|---|
| trivia00 | false | `The capital of Australia is **Canberra**.` |
| trivia01 | false | `The novel **Pride and Prejudice** was written by **Jane Austen**.` |
| trivia02 | false | `The chemical symbol for gold is **Au**.` |
| trivia03 | false | `The Berlin Wall itself didn't "fall" in a single event, but the **opening and dismantling …` |
| trivia04 | false | `The largest planet in the Solar System is **Jupiter**.` |

Оборванные (`truncated=true`, строки **не пропущены**, `status=ok`):

| id | n_tokens | ответ (хвост отрезан бюджетом) |
|---|---|---|
| trivia09 | 81 | `The answer to what is the **longest river in the world** can be debated depending on whether you consider the longest by…` |
| trivia11 | 83 | `No one has ever walked on the Moon. However, **Astronauts** have been able to travel to the Moon and perform spacewalks…` |
| trivia14 | 79 | `That's a philosophical question, as "hardest" can be interpreted in several ways (density, hardness, or resistance). If…` |
| trivia19 | 76 | `The discovery of penicillin is generally credited to **Alexander Fleming** in **1928**. Fleming observed that a mold (P…` |

Сквозные поля в `meta.jsonl` (row 0):
`{"id":"trivia00", "gold":"Canberra", "dataset":"triviaqa-dev", "split":0, …}` — все ключи входа, кроме `id`/`question`/`answer`, скопированы как есть.

Тот же прогон с `--thinking`: truncated **19/20**, промпт длиннее (22–25 токенов), 20.0 s, 1.00 ex/s — ожидаемо, т.к. thinking-режим упирается в бюджет токенов.

---

## 8. Адаптация под Qwen3.5-0.8B

Модель: `unsloth\Qwen3.5-0.8B-GGUF\Qwen3.5-0.8B-BF16.gguf` (1.4 ГБ), `n_layer=24`, `n_embd=1024`.
Архитектура гибридная: **полное внимание каждым 4-м слоем** (3, 7, 11, 15, 19, 23 — тензоры
`Qcur-*`, `Kcur-*`, `Vcur-*`, `__fattn__-N`), остальные слои — **рекуррентное (линейное)
внимание** (`conv_states-N`, `cache_r_lN`, `cache_s_lN`, `__fgdn_ch__-N`, `linear_attn_out-N`).

### 8.1. Разведка: `--dump-tensors` (до правок)

В лог дампа добавлена только форма тензора для `l_out-*` (одна строка, поведение не меняется).
Прогоны: одиночный режим на 8- и 43-токенном промпте, пакетный режим на 5 и на 20 примерах.

| Что | Результат |
|---|---|
| какие `l_out-N` есть | `l_out-0 … l_out-23` — **по одному тензору на каждый из 24 слоёв**, пропусков нет |
| сколько строк | `[1024 x n_tokens]` **у всех слоёв, включая последний** — строки не урезаны до числа выходов |
| тип | `f32` во всех 29 904 чтениях прогона на Qwen (ветка BF16 в `read_row` не понадобилась) |
| v2, 5 примеров | `[1024 x 22]`, `[1024 x 20]`, ... — тоже полное число строк |

Последний слой `l_out-23` ведёт себя как обычный: строк `n_tokens`, а не `n_outputs`.
Фолбэк «строки по порядку логитов» (см. 8.2) реализован на случай иных бэкендов/моделей,
на Qwen он не срабатывал.

### 8.2. Правки кода

1. **Захват по слоям.** Вместо флагов `got_c`/`got_a` — массивы `cap_c`/`cap_a` по слоям и
   счётчики `n_cap_c`/`n_cap_a` (захваченный слой считается один раз за проход). После
   `llama_decode` требуется `captured == L_sel`, иначе
   `skip("incomplete capture: x/L")`. В одиночном режиме — `LOG_ERR` и код 1.
2. **Фолбэк строк.** Если у `l_out-N` строк `<= pos_c` (остались только выходы), строки
   берутся по порядку запроса логитов: `A -> 0`, `C -> 1`, при `A == C` — строка `0`.
   Тот же код используется в одиночном режиме.
3. **Шаблон Qwen.** `enable_thinking=false` (поведение по умолчанию) — блока `think` в
   отрендеренном промпте нет; промпт первой строки печатается в лог один раз.
4. **Baseline-статистика генерации.** В `meta.jsonl` для строк с `--generate` добавлены
   `logprobs` (log p выбранного токена) и `entropies` (энтропия полного распределения),
   обычный softmax по «сырым» логитам без температуры, float32 (аккумуляторы double).
   Для строк с готовым ответом (без генерации) — `null`.
5. **Id генерации.** При `--generate` в проход состояний идут **сохранённые id** напрямую.
   Повторная токенизация текста ответа считается отдельно, выводится доля расхождений
   (`retokenize mismatch: x/y`).

### 8.3. Проверки

| Проверка | Результат |
|---|---|
| **Регрессия Gemma E2B**, те же 20 вопросов и те же флаги (`--generate --max-new-tokens 64 --max-len 1024 --layers step=2`) | `hidden_a.npy` / `hidden_c.npy` **байт-в-байт** совпали с прогоном до правок (SHA-256); 20/20 ok, truncated=4; `retokenize mismatch 0/20` |
| **Qwen3.5-0.8B**, 20 вопросов TriviaQA-стиля, `--generate --max-new-tokens 64 --layers all` | 20/20 ok, 0 skipped; захвачено слоёв **24/24** на каждом примере (проверка `captured == L_sel` прошла, иначе строка ушла бы в `skipped` с `incomplete capture: 24/24`) |
| truncated на Qwen | **15/20 = 75 %** |
| формы массивов | `hidden_a` / `hidden_c` = **(20, 24, 1024)** float16 |
| NaN / Inf | нет |
| последний слой | непустой (`any != 0`), **не равен** предыдущему (`max abs(l23 - l22) = 3.49`) |
| отрендеренный промпт Qwen (thinking выключен) | `user`-роль с вопросом, **без блока `think`** |
| **Порядок примеров** (item 3) | прямой и обратный порядки тех же 20 вопросов: `hidden_a`, `hidden_c`, ответы и `logprobs` совпали **по каждому id** (0 расхождений из 20) |
| `logprobs` / `entropies` | заполнены у всех 20 сгенерированных строк; длина списков = числу сгенерированных токенов (row0: 61 = `n_tokens` 80 - `pos_a` 18 - 1); диапазоны: logprobs `[-2.77, 0.00]`, entropies `[0.00, 4.80]` |
| ошибки в логе Qwen (29 904 чтения `l_out`, полный прогон 20 примеров) | `ERR` / `unsupported` / `non-finite` / `incomplete capture` — **нет** |

5 ответов Qwen (`--max-new-tokens 64`, `--layers all`):

| id | truncated | ответ |
|---|---|---|
| trivia00 | false | `The capital of Australia is **Canberra**.` + продолжение |
| trivia01 | true | `The novel *Pride and Prejudice* was written by **Jane Austen**.` + продолжение |
| trivia02 | true | `The chemical symbol for gold is **Au**.` + продолжение |
| trivia03 | true | `The Berlin Wall fell on **November 9, 1989**.` + продолжение |
| trivia04 | true | `The largest planet in the Solar System is **Jupiter**.` + продолжение |

Высокая доля `truncated` на Qwen ожидаема: модель отвечает коротко, но затем продолжает
развёрнутый текст и упирается в `--max-new-tokens 64`.

---

## 9. Данные: подготовка и пилот

Каталог `pipeline/` — Python-скрипты подготовки данных (без модели, только `uv` + `datasets`).

### 9.1. Источник

- Датасет: `mandarjoshi/trivia_qa`, конфиг `rc.nocontext`, сплит `validation` (17 944 строк).
- Поля совпали с карточкой датасета: `question_id`, `question`, `answer.value`, `answer.aliases`.
- Кэш Hugging Face — по умолчанию.

### 9.2. Фильтры и сплиты

`prepare_data.py` (шаги в порядке выполнения):

1. Дедупликация по `normalize(question)`, остаётся первый: **−7 983** дублей (17 944 → 9 961).
2. Отброс: вопросы длиннее 300 символов и без эталона (`answer.value` пуст): 9 961 → **9 947**.
3. Группа = `normalize(answer.value)`; **7 400 групп**.
4. Сплит по группам, детерминированно: `sha256(f"{seed}:{group}") % 100` → `<60` train, `<80` val, иначе test (60/20/20). Все вопросы с одним эталоном — в одном сплите.
5. Случайный сэмпл `--n` с сохранением долей сплитов (квоты `round(n*0.6)`, `round(n*0.2)`, остаток test).

`normalize.py` — общий модуль: нижний регистр, удаление пунктуации и артиклей (`a`/`an`/`the`), схлопывание пробелов. Используется здесь и будет в будущем `label.py`.

### 9.3. Выход

Шаблон вопроса (константа): `"{question}\nAnswer with just the answer, no explanation."`

JSONL, строка на вопрос:

```json
{"id": "...", "question": "<шаблон>", "gold": ["norm(value)", "norm(alias)", ...],
 "gold_main": "<value>", "dataset": "triviaqa-rc-nocontext-validation",
 "split": "train|val|test", "group": "<norm(value)>"}
```

`--meta` (JSON): `seed`, `n`, имя датасета/конфиг/сплит, шаблон, числа по сплитам, число групп, число удалённых дублей, SHA-256 списка id.

### 9.4. Пилот: `--n 200`

```
python prepare_data.py --n 200 --out data\pilot.jsonl --meta data\pilot_meta.json
```

| Что | Значение |
|---|---|
| строк после фильтров | 9 947 |
| дублей удалено | 7 983 |
| групп | 7 400 |
| сплиты (сэмпл) | train=120, val=40, test=40 |
| ids sha256 | `86ca977a565b88c0fce0d9936b27a927bda5dc705b831ec6a0721a6abb1972f5` |

Прогон инструмента на Qwen3.5-0.8B-BF16 (`--generate --max-new-tokens 64 --max-len 1024 --layers all -ngl 99`):

| Что | Значение |
|---|---|
| ok / skipped | 200 / 0 |
| truncated | **0 (0 %)** |
| скорость | **5.31 ex/s** (37.7 s) |
| непустых ответов | 200 (100 %) |
| retokenize mismatch | 1/200 (0.5 %) |
| сквозные поля в `meta.jsonl` | `gold`, `split`, `group` — на месте; распределение сплитов 120/40/40 |

**Разница с TriviaQA-20 из раздела 7.1:** там truncated 4/20 (Gemma) и 15/20 (Qwen) — вопросы без подсказки «just the answer» вели к развёрнутым ответам. Здесь шаблон явно ограничивает формат, поэтому **ни один из 200 ответов не упёрся в лимит токенов**, средняя скорость выросла с ~0.9 до 5.3 ex/s.

Пример (10 случайных, `gold_main` vs ответ): модель часто отвечает коротко и по делу (`Boreas` → `Nereus`, `1` → `1`), но встречаются и ошибки (`JOY DIVISION` → `The Beatles`, `Alan Freed` → `Elvis Presley`) — оценка правильности (label.py) — следующий шаг.
