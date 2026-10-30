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

---

## 10. Разметка правильности: `label.py`

`pipeline/label.py` — разметка правильности ответов из `meta.jsonl` (вывод `llama-hidden-states`) по эталонам `gold`. Две метки: `correct_strict` (строгое правило) и `correct_relaxed` (ослабленное). Колонка `correct` оставлена для совместимости = `correct_strict`.

### 10.1. Нормализация с запасным вариантом

`normalize()` из `normalize.py` удаляет пунктуацию и артикли (`a`/`an`/`the`), поэтому ответ или эталон, состоящий только из артикла/пунктуации, нормализуется в пустую строку. В `label.py` добавлено правило: **если `normalize(x)` пусто, используется запасная нормализация** (нижний регистр, удаление пунктуации, **без** удаления артиклей). Применяется и к ответу, и к каждому `gold`. Благодаря этому строки **больше не уходят в `null`** по причине пустой нормализации (единственная причина `null` теперь — `truncated=true` либо вовсе пустой ответ, либо строка не `ok`/не `generated`).

### 10.2. Метки

Для строк `status=ok` и `generated=true` (и не `truncated`):

- **`correct_strict`** = 1, если **любой** `gold` встречается в `ans` **как целая фраза по границам слов** (сравнение токенов, не подстрок): последовательность токенов `gold` равна непрерывной подпоследовательности токенов `ans`. `"1"` не совпадает с `"1990s"`/`"10"`; `"web link"` не совпадает с `"web links"`.
- **`correct_relaxed`** = 1, если выполняется **любое** из:
  - `correct_strict` (то же строгое правило);
  - у `gold` ≥ 2 слов и **все** слова `gold` встречаются в ответе как целые слова (**порядок не важен**);
  - совпадение после **простой нормализации числа** (удаление конечного `s` у слов длиной ≥ 4; короткие слова не трогаются: `web links` → `web link`), затем строгое фразовое совпадение.

`correct_relaxed` всегда ≥ `correct_strict`. `correct = correct_strict`.

### 10.3. Выход

CSV (`--out`), колонки: `id, row, split, correct_strict, correct_relaxed, correct, truncated, n_answer_tokens, answer, gold_main`. `n_answer_tokens = pos_c - pos_a`.

В консоль: число строк по сплитам, обе доли `correct` (общая и по сплитам), число `null`, число строк, где `strict != relaxed`. Предупреждение, если строгая доля вне 10–90 %.

### 10.4. Аудит

- `--audit N --seed S` — выборка `N` случайных **размеченных** строк в `audit.csv` (колонки `answer, gold_main, correct_strict, correct_relaxed, manual`; `manual` пустая — для ручной разметки 0/1);
- `--audit-score audit.csv` — для **каждой** из двух меток: доля совпадения с ручной, число ложных срабатываний (auto=1, manual=0) и ложных пропусков (auto=0, manual=1), доля шума `(fp+fn)/total`; плюс итоговая доля шума (среднее двух).

### 10.5. Пилот (200 вопросов, Qwen3.5-0.8B)

```
python label.py --meta out\pilot_qwen08\meta.jsonl --out labels.csv
```

| split | строк | strict | доля | relaxed | доля |
|---|---|---|---|---|---|
| train | 120 | 25/120 | 0.208 | 25/120 | 0.208 |
| val | 40 | 4/40 | 0.100 | 5/40 | 0.125 |
| test | 40 | 9/40 | 0.225 | 10/40 | 0.250 |
| **overall** | **200** | **38/200** | **0.190** | **40/200** | **0.200** |

`null` = 0 (запасная нормализация спасла строку с ответом `"A"`). `strict != relaxed`: **2 строки**.

Строки, где метки расходятся:

| id | gold_main | answer | strict | relaxed | почему relaxed=1 |
|---|---|---|---|---|---|
| jp_3910 | Rothschild | `The Rothschilds` | 0 | 1 | нормализация числа: `rothschilds` → `rothschild` |
| bb_3197 | Germany, Japan, Italy | `Germany, Italy, and Japan` | 0 | 1 | ≥2 слов, все слова есть, порядок не важен |

### 10.6. Примеры (строгая метка)

`correct_strict=1` (10 случайных):

| gold_main | answer |
|---|---|
| Utrecht | `Treaty of Utrecht` |
| Heraldry | `Heraldry` |
| Les Paul | `Gibson Guitars named after him: Les Paul` |
| Eye | `eye` |
| Winnipeg | `Winnipeg` |
| Printed Circuit Board | `Printed Circuit Board` |
| Chess | `Chess` |
| THE AVENGERS | `The Avengers` |
| PULSAR | `pulsar` |
| 1 | `1` |

`correct_strict=0` (10 случайных):

| gold_main | answer |
|---|---|
| Miller Lite Beer | `Burger` |
| TOP OF THE POPS | `The Wizard` |
| ALMONDS | `hydrogen cyanide` |
| Germany, Japan, Italy | `Germany, Italy, and Japan` |
| Sea | `ocean` |
| APOLLO | `King Philip II of Greece` |
| Royal Botanical Gardens at Kew | `The 250th anniversary of the Royal Mint.` |
| London | `The Sea Wolf was written by John F. Kennedy.` |
| A Horse Collar | `a necklace` |
| Joan Rivers | `Maggie Gyllenhaal` |

Замечания:

- `strict != relaxed` всего на 2 строках из 200 — ослабленное правило заметно не завышает долю (0.190 → 0.200);
- ответ с **другим порядком компонентов** не засчитывается строго, но засчитывается relaxed (порядок не важен);
- ответ, содержащий эталон **как фразу внутри** более длинного текста (`Treaty of Utrecht`), засчитывается строго — это ожидаемо для оценки «содержит эталон»;
- доля `correct` ≈ 19–20 % на пилоте низкая, но ожидаемо для 0.8B-модели на TriviaQA без контекста; большинство ошибок — фактические ошибки модели, а не артефакты разметки.

---

## 11. Основной прогон (6000 вопросов, Qwen3.5-0.8B)

### 11.1. Данные

```
python prepare_data.py --n 6000 --out data\main.jsonl --meta data\main_meta.json
```

| Что | Значение |
|---|---|
| строк после фильтров | 9 947 |
| дублей удалено | 7 983 |
| групп | 7 400 |
| сплиты (сэмпл) | train=3600, val=1200, test=1200 |
| ids sha256 | `6ee7ff7e1524bf902e141fe373509d929aa2125f787404211d06345093af153b` |
| `data\main.jsonl` | 3 128.8 KiB |

### 11.2. Прогон инструмента

```
llama-hidden-states.exe -m Qwen3.5-0.8B-BF16.gguf --batch data\main.jsonl \
  --generate --max-new-tokens 64 --max-len 1024 --layers all -ngl 99 \
  --out out\main_qwen08
```

| Что | Значение |
|---|---|
| время | **1287.1 s** |
| скорость | **4.66 ex/s** |
| ok / skipped | 6000 / 0 |
| truncated | **23 (0.4 %)** |
| пустых ответов | 0 |
| `meta.jsonl` | 4.5 MiB |
| `hidden_a.npy` / `hidden_c.npy` | по 281.3 MiB |

### 11.3. Разметка (`label.py`)

```
python label.py --meta out\main_qwen08\meta.jsonl --out out\main_qwen08\labels.csv
python label.py --meta out\main_qwen08\meta.jsonl --audit 100 --seed 2 --out out\main_qwen08\audit.csv
```

В `audit.csv` добавлены все строки, где `strict != relaxed` (23 шт.), итого **123 строки**.

| split | строк | labeled | null | strict | доля | relaxed | доля | diff |
|---|---|---|---|---|---|---|---|---|
| train | 3600 | 3589 | 11 | 597/3589 | 0.166 | 610/3589 | 0.170 | 13 |
| val | 1200 | 1193 | 7 | 193/1193 | 0.162 | 200/1193 | 0.168 | 7 |
| test | 1200 | 1195 | 5 | 214/1195 | 0.179 | 217/1195 | 0.182 | 3 |
| **total** | **6000** | **5977** | **23** | **1004/5977** | **0.168** | **1027/5977** | **0.172** | **23** |

- `null` = 23 = число `truncated` (запасная нормализация устранила прочие null);
- `strict != relaxed` = 23 строки (все ушли в `audit.csv`);
- `labels.csv` = 431.4 KiB, `audit.csv` = 6.0 KiB.

Доля правильности ≈ 17 % стабильна по сплитам (0.162–0.179) — сплит по группам сбалансирован. Ослабленное правило добавляет лишь +0.004 (0.168 → 0.172).

## 12. Baseline: logprob и энтропия первого токена

Шаг 4 `GLOBALPLAN.md`. Код — `pipeline/baseline.py` (только `numpy`); AUROC/AUPRC/Brier/ECE
реализованы в нём же и сверены с брутфорс-версиями (включая обработку совпадающих скоров).

**Данные.** 5977 строк с непустой меткой `correct_filtered_b`; 23 `truncated` исключены.
Стыковка `meta.jsonl` и `labels.csv` — по `row` (в `labels.csv` есть дубль `id`
`bb_956`: две разные строки с разными вопросами).

| split | строк | positive | доля |
|---|---|---|---|
| val | 1193 | 166 | 0.139 |
| test | 1195 | 178 | 0.149 |

**Признаки.** Значения в первом токене ответа: `logprobs[0]` (логарифм вероятности
выбранного токена; выше — увереннее) и `entropies[0]` (энтропия распределения; ниже —
увереннее). Скор для AUROC/AUPRC — `lp0` и `−h0`; калибровка на них не влияет, так как
монотонна.

**Калибровка.** На val подгоняется `p = sigmoid(b0 + b1·x)` (IRLS), отдельно для
`x = −lp0` и `x = −h0`; на test эти коэффициенты применяются без переобучения.

| признак | b0 | b1 | итоговая формула |
|---|---|---|---|
| `logprob` | −0.0111 | −1.0691 | p ≈ sigmoid(1.069·lp0) |
| `entropy` | 0.5970 | 0.5501 | p = sigmoid(0.597 − 0.550·h0) |

Калиброванные метрики на val — in-sample (коэффициенты подогнаны на тех же строках),
поэтому честная оценка качества калибровки — только test.

**Результаты** (bootstrap 1000 ресэмплов, seed 42; ECE — 10 равных бинов):

| split | признак | n | pos | base | AUROC [95 % CI] | AUPRC [95 % CI] | Brier [95 % CI] | ECE [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| val | logprob | 1193 | 166 | 0.139 | 0.729 [0.689, 0.770] | 0.319 [0.258, 0.397] | — | — |
| val | entropy | 1193 | 166 | 0.139 | 0.743 [0.703, 0.785] | 0.336 [0.275, 0.416] | — | — |
| val | logprob, калибр. | 1193 | 166 | 0.139 | 0.729 [0.689, 0.770] | 0.319 [0.258, 0.397] | 0.109 [0.097, 0.121] | 0.012 [0.008, 0.034] |
| val | entropy, калибр. | 1193 | 166 | 0.139 | 0.743 [0.703, 0.785] | 0.336 [0.275, 0.416] | 0.107 [0.094, 0.120] | 0.015 [0.011, 0.038] |
| val | сырая `exp(lp0)` | 1193 | 166 | 0.139 | 0.729 [0.689, 0.770] | 0.319 [0.258, 0.397] | 0.118 [0.106, 0.131] | 0.063 [0.046, 0.081] |
| test | logprob | 1195 | 178 | 0.149 | 0.739 [0.700, 0.775] | 0.321 [0.266, 0.393] | — | — |
| test | entropy | 1195 | 178 | 0.149 | 0.767 [0.727, 0.803] | 0.350 [0.292, 0.422] | — | — |
| test | logprob, калибр. | 1195 | 178 | 0.149 | 0.739 [0.700, 0.775] | 0.321 [0.266, 0.393] | 0.114 [0.102, 0.127] | 0.014 [0.009, 0.036] |
| test | entropy, калибр. | 1195 | 178 | 0.149 | 0.767 [0.727, 0.803] | 0.350 [0.292, 0.422] | 0.112 [0.099, 0.124] | 0.025 [0.016, 0.046] |
| test | сырая `exp(lp0)` | 1195 | 178 | 0.149 | 0.739 [0.700, 0.775] | 0.321 [0.266, 0.393] | 0.124 [0.113, 0.136] | 0.060 [0.045, 0.080] |

**Выводы.**

- Baseline заметно выше случайного: AUROC 0.73–0.77, AUPRC 0.32–0.35 при доле
  положительных 0.139–0.149. Это точка отсчёта для пробы (шаг 5).
- Энтропия первого токена чуть сильнее logprob (test: AUROC 0.767 vs 0.739, AUPRC
  0.350 vs 0.321), но интервалы перекрываются — различие не установлено.
- Сырая вероятность `exp(lp0)` систематически переоценена (ECE 0.060–0.063);
  калибровка на val снижает ECE до 0.012–0.025 и Brier с 0.118 до 0.109 (val) и
  с 0.124 до 0.114 (test). То есть logprob полезен как ранжирование, но как
  «вероятность правильности» требует калибровки.

**Воспроизведение и артефакты.**

```
python baseline.py --meta out\main_qwen08\meta.jsonl --labels out\main_qwen08\labels.csv \
  --out out\main_qwen08\baseline.json --scores-out out\main_qwen08\baseline_scores.csv
```

- `out/main_qwen08/baseline.json` — коэффициенты калибровки, счётчики, метрики с интервалами;
- `out/main_qwen08/baseline_scores.csv` — по строкам: `id, row, split, y, lp0, h0,
  p_raw, p_cal_logprob, p_cal_entropy` (5977 строк, только val/test).

## 13. Проба: логистическая регрессия по скрытым состояниям (шаг 5)

Код — `pipeline/probe.py`. Признак — весь вектор скрытого состояния слоя `L` в точке
A или C (1024 числа), стандартизованный по train. Модель — L2-логистическая регрессия
(Ньютон/IRLS, шаг ограничен по норме).

**Протокол выбора (зафиксирован до вычисления test).**

1. `L2`-сила `λ` — из фиксированной сетки `{100, 300, 1000, 3000}`; выбирается по
   **среднему val AUROC по всем 48 конфигам** (24 слоя × 2 точки), а не по максимуму,
   чтобы не подгоняться под шум одной конфигурации.
2. Конфигурация (точка, слой) — по максимуму val AUROC при выбранном `λ`.
3. Test считается **один раз** для выбранной конфигурации (плюс объявленные заранее
   контроли).

Данные: train 3589 (512 положительных), val 1193 (166), test 1195 (178); метка
`correct_filtered_b`, 23 `truncated` исключены.

**Выбор `λ`.**

| λ | средний val AUROC по 48 конфигам |
|---|---|
| 100 | 0.7905 |
| 300 | 0.8034 |
| **1000** | **0.8085** |
| 3000 | 0.8038 |

**Кривая val AUROC по слоям при `λ = 1000`** (train AUROC для диагностики
переобучения — в `probe_curve.csv`):

| слой | A | C |
|---|---|---|
| 0 | 0.689 | 0.717 |
| 1 | 0.708 | 0.766 |
| 2 | 0.708 | 0.757 |
| 3 | 0.701 | 0.763 |
| 4 | 0.715 | 0.803 |
| 5 | 0.733 | 0.808 |
| 6 | 0.738 | 0.820 |
| 7 | 0.754 | 0.831 |
| 8 | 0.761 | 0.852 |
| 9 | 0.790 | 0.861 |
| 10 | 0.802 | 0.871 |
| 11 | 0.809 | 0.874 |
| 12 | 0.808 | 0.884 |
| **13** | 0.804 | **0.890** |
| 14 | 0.808 | 0.883 |
| 15 | 0.813 | 0.875 |
| 16 | 0.822 | 0.865 |
| 17 | **0.826** | 0.867 |
| 18 | 0.826 | 0.862 |
| 19 | 0.818 | 0.858 |
| 20 | 0.813 | 0.856 |
| 21 | 0.813 | 0.859 |
| 22 | 0.810 | 0.856 |
| 23 | 0.807 | 0.855 |

Выбранная конфигурация: **точка C, слой 13** (val AUROC 0.8895, val AUPRC 0.6345).

**Метрики выбранной конфигурации** (bootstrap 1000, seed 42):

| split | AUROC [95 % CI] | AUPRC [95 % CI] | Brier | ECE |
|---|---|---|---|---|
| val | 0.890 [0.861, 0.915] | 0.635 | 0.081 | 0.035 |
| **test** | **0.857 [0.825, 0.885]** | **0.593 [0.523, 0.662]** | **0.089** | **0.019** |
| test, Platt по val | 0.857 | 0.593 | 0.091 | 0.031 |

Platt-калибровка (`p = sigmoid(0.734 + 1.399·score)`, подгонка на val) test не улучшает:
исходные вероятности пробы уже близки к калиброванным.

**Контроли.**

| контроль | val AUROC | test AUROC |
|---|---|---|
| перемешанные метки train (5 перемешиваний, seed 42) | 0.443–0.570 | **0.490 ± 0.052** |
| слой 0, точка A | 0.689 | 0.681 |
| слой 0, точка C | 0.717 | 0.718 |

Перемешанные метки дают ~0.5 — процедура не «подсматривает»; слой 0 заметно слабее
выбранного 13-го, то есть рост идёт с глубиной, а не от одного лишь формата входа.

**Интерпретация.**

- Проба уверенно побивает baseline: test AUROC 0.857 против 0.739 (logprob) и 0.767
  (энтропия); AUPRC 0.593 против 0.321 и 0.350. При доле правильных 0.149 это уже
  рабочее ранжирование.
- Разрыв val (0.890) → test (0.857) ожидаем: конфигурация выбиралась по val, поэтому
  честная оценка — только test; разрыв 0.033 — цена выбора из 48 конфигов.
- На val точка C сильнее A в лучшей точке (0.890 на слое 13 против 0.826 на слое 17),
  но тест для A не считался — это осталось за протоколом «тест один раз».
- У C качество растёт до 13–14 слоя и затем падает; у A — до 17–18. У Qwen слои
  чередуются (полное и линейное внимание), поэтому «лесенка» в кривой ожидаема.

**Артефакты.**

```
python probe.py --out out\main_qwen08\probe.json \
  --curve-out out\main_qwen08\probe_curve.csv \
  --scores-out out\main_qwen08\probe_scores.csv
```

- `probe.json` — протокол, выбор λ, полная кривая (192 строки), метрики, контроли;
- `probe_curve.csv` — `point, layer, lambda, train_auroc, val_auroc, val_auprc`;
- `probe_scores.csv` — `id, row, split, y, score, prob, prob_platt` (val + test выбранной
  конфигурации).

## 14. Вторая модель: Gemma 4 E2B (шаг 7)

Модель: `gemma-4-E2B-it-Q4_K_M.gguf` (35 слоёв, D=1536). Тот же протокол, что для Qwen:
тот же вход `data/main.jsonl`, шаблон, `--generate --max-new-tokens 64 --max-len 1024
--layers all -ngl 99`.

### 14.1. Пилот 200 и полный прогон

| Что | Пилот (`out/pilot_gemma4e2b`) | Полный (`out/main_gemma4e2b`) |
|---|---|---|
| примеров | 200 | 6000 |
| ok / skipped | 200 / 0 | 6000 / 0 |
| `truncated` | 0 | **13 (0.2 %)** |
| время | 40.6 s | **1258.1 s** |
| скорость | 4.93 ex/s | **4.77 ex/s** |
| retokenize mismatch | 0/200 | 32/6000 (0.5 %) |
| массивы | `[200, 35, 1536]` | `[6000, 35, 1536]` |
| размер `.npy` | по 21.5 МБ | по 645.1 МБ |

Скорость и обрезки на порядок отличаются от прежнего теста на `in_trivia20.jsonl`
(2.12 ex/s, truncated 20 %): там были синтетические вопросы без шаблона короткого
ответа, и модель отвечала развёрнуто. На `main.jsonl` с шаблоном
`Answer with just the answer, no explanation.` ответы короткие: медиана 3 токена,
максимум 29 на пилоте и 64 на полном прогоне.

### 14.2. Особенности ответов Gemma

- **25 строк без метки:** 13 `truncated` + 12 нетривиальных. Из 12: 9 — генерация
  нулевой длины (0 токенов, сразу EOG), 3 — вырожденные ответы `}`, `?`, `\r`.
  У Qwen таких не было (все 23 null — ровно `truncated`).
- **4 из 13 обрезанных строк начинаются с `<|channel>thought\nThinking Process:`** —
  утечка thinking-канала, хотя `config.json` пишет `thinking: false`. Причина не
  исследована.
- В остальных обрезанных строках — другой мусор: `3333…`, добивка неразрывными
  пробелами `\xa0` (в одной строке ответ целиком из них), HTML `<strong>`, обрывок JSON,
  развёрнутая проза вместо ответа.
- **Первый токен почти всегда служебный:** медиана `logprobs[0]` = −0.235, у 39.8 %
  строк `lp0 > −0.1` (вероятность > 0.90). У Qwen — 1.3 %. Поэтому сырая вероятность
  `exp(lp0)` для Gemma калибрована катастрофически (ECE 0.518), хотя ранжирует она
  всё же лучше случайного.

### 14.3. Разметка

| правило | единиц / 5975 | доля |
|---|---|---|
| strict | 1211 | 0.203 |
| relaxed | 1232 | 0.206 |
| filtered_a | 1074 | 0.180 |
| **filtered_b (итоговое)** | **1087** | **0.182** |

Сплиты: train 677/3588 = 0.189, val 215/1193 = 0.180, test 195/1194 = 0.163.
Фильтр `a` меняет 137 меток, `b` — 124, между собой они расходятся на 13 строках.
Общая доля правильных у Gemma выше, чем у Qwen (0.182 против 0.143).

### 14.4. Baseline Gemma

| признак | split | AUROC | AUPRC | Brier | ECE |
|---|---|---|---|---|---|
| logprob | val | 0.689 | 0.377 | — | — |
| logprob | test | 0.705 | 0.360 | — | — |
| entropy | val | 0.699 | 0.384 | — | — |
| entropy | test | 0.712 | 0.367 | — | — |
| logprob, калибр. | test | 0.705 | 0.360 | 0.131 | 0.018 |
| entropy, калибр. | test | 0.712 | 0.367 | 0.130 | 0.017 |
| сырая `exp(lp0)` | test | 0.705 | 0.360 | 0.464 | 0.540 |

Baseline у Gemma слабее, чем у Qwen (test AUROC 0.705–0.712 против 0.739–0.767), а
сырая вероятность бесполезна совсем.

### 14.5. Проба Gemma

Сетка `λ` та же: средние val AUROC 100 → 0.8193, 300 → 0.8343, **1000 → 0.8420**,
3000 → 0.8409. Выбрано `λ = 1000`, **точка C, слой 19**.

**Кривая val AUROC по слоям (λ = 1000):**

| слой | A | C |
|---|---|---|
| 0 | 0.739 | 0.778 |
| 1 | 0.767 | 0.800 |
| 2 | 0.771 | 0.799 |
| 3 | 0.773 | 0.816 |
| 4 | 0.780 | 0.835 |
| 5 | 0.782 | 0.848 |
| 6 | 0.781 | 0.858 |
| 7 | 0.791 | 0.862 |
| 8 | 0.792 | 0.878 |
| 9 | 0.785 | 0.889 |
| 10 | 0.796 | 0.900 |
| 11 | 0.806 | 0.898 |
| 12 | 0.802 | 0.900 |
| 13 | 0.799 | 0.900 |
| 14 | 0.801 | 0.907 |
| 15 | 0.807 | 0.898 |
| 16 | 0.812 | 0.898 |
| 17 | 0.818 | 0.897 |
| 18 | 0.827 | 0.900 |
| **19** | 0.826 | **0.908** |
| 20 | 0.829 | 0.905 |
| 21 | 0.823 | 0.903 |
| 22 | 0.823 | 0.903 |
| 23 | 0.828 | 0.903 |
| 24 | 0.825 | 0.898 |
| 25 | 0.830 | 0.887 |
| 26 | 0.836 | 0.880 |
| 27 | 0.836 | 0.878 |
| 28 | 0.833 | 0.877 |
| 29 | 0.827 | 0.874 |
| 30 | 0.828 | 0.872 |
| 31 | 0.830 | 0.873 |
| 32 | 0.838 | 0.871 |
| **33** | **0.840** | 0.876 |
| 34 | 0.828 | 0.870 |

| split | AUROC [95 % CI] | AUPRC [95 % CI] | Brier | ECE |
|---|---|---|---|---|
| val | 0.908 [0.886, 0.928] | 0.709 | 0.088 | 0.023 |
| **test** | **0.935 [0.918, 0.950]** | **0.744 [0.681, 0.804]** | **0.073** | **0.029** |

Platt-калибровка не меняет ничего (Brier 0.0735, ECE 0.0294).

Контроли: перемешанные метки — test AUROC 0.521 ± 0.056; слой 0 — A 0.714, C 0.780.

Отличие от Qwen: у Gemma тест **выше** val (0.935 против 0.908), тогда как у Qwen
тест ниже (0.857 против 0.890). Причина не установлена; val и test — разные наборы
вопросов (доля правильных 0.180 против 0.163), и на Gemma разброс между сплитами
оказался в другую сторону.

### 14.6. Сравнение Qwen и Gemma

| | Qwen3.5-0.8B | Gemma 4 E2B |
|---|---|---|
| слоёв × D | 24 × 1024 | 35 × 1536 |
| правильных (итог, filtered_b) | 0.143 | 0.182 |
| baseline logprob, test AUROC | 0.739 | 0.705 |
| baseline entropy, test AUROC | 0.767 | 0.712 |
| проба: λ | 1000 | 1000 |
| проба: точка, слой | C, 13 | C, 19 |
| проба: val AUROC | 0.890 | 0.908 |
| **проба: test AUROC** | **0.857** [0.825, 0.885] | **0.935** [0.918, 0.950] |
| проба: test AUPRC | 0.593 | 0.744 |
| control: слой 0 (C), test | 0.718 | 0.780 |
| control: перемешанные метки | 0.490 ± 0.052 | 0.521 ± 0.056 |

Относительная глубина лучшего слоя почти совпала: 13/24 = 0.54 у Qwen и 19/35 = 0.54
у Gemma, точка C лучше A у обеих.

**Парное сравнение на одних и тех же вопросах** (`compare_probes.py`, bootstrap 1000,
ресэмплинг вопросов; у каждой модели свои метки, поэтому метки расходятся на 180 из
1189 строк):

| | значение | 95 % CI |
|---|---|---|
| общих тестовых строк | 1189 (Qwen 0.149, Gemma 0.164 положительных) | — |
| AUROC Qwen / Gemma | 0.858 / 0.936 | — |
| ΔAUROC (Gemma − Qwen) | **+0.078** | [+0.048, +0.106] |
| ΔAUPRC (Gemma − Qwen) | **+0.151** | [+0.080, +0.225] |

Все 1000 bootstrap-ресэмплов дали преимущество Gemma. Но это **не чистое сравнение
моделей**: метка у каждой модели своя (правильность её собственного ответа), и доля
правильных у Gemma выше (0.182 против 0.143). Разница отражает и то, что Gemma чаще
отвечает верно, и то, что её состояния лучше отделяют верное от неверного.

Пересечение меток на 5952 общих строках: обе верны 479, только Qwen 373, только
Gemma 606, обе неверны 4494.

**Артефакты.**

```
llama-hidden-states.exe -m gemma-4-E2B-it-Q4_K_M.gguf --batch data\main.jsonl \
  --generate --max-new-tokens 64 --max-len 1024 --layers all -ngl 99 --out out\main_gemma4e2b

python label.py --meta out\main_gemma4e2b\meta.jsonl --out out\main_gemma4e2b\labels.csv
python baseline.py --meta out\main_gemma4e2b\meta.jsonl --labels out\main_gemma4e2b\labels.csv \
  --out out\main_gemma4e2b\baseline.json --scores-out out\main_gemma4e2b\baseline_scores.csv
python probe.py --hidden-a out\main_gemma4e2b\hidden_a.npy --hidden-c out\main_gemma4e2b\hidden_c.npy \
  --meta out\main_gemma4e2b\meta.jsonl --labels out\main_gemma4e2b\labels.csv \
  --out out\main_gemma4e2b\probe.json --curve-out out\main_gemma4e2b\probe_curve.csv \
  --scores-out out\main_gemma4e2b\probe_scores.csv
python compare_probes.py --a out\main_qwen08\probe_scores.csv \
  --b out\main_gemma4e2b\probe_scores.csv --split test \
  --out out\compare_qwen08_gemma4e2b_test.json
```

## 15. Третья модель: gpt-oss-20b (шаг 8)

Модель: `gpt-oss-20b-MXFP4.gguf` (12.1 ГБ, архитектура `gpt-oss` = `LLM_ARCH_OPENAI_MOE`,
**24 слоя**, **D = 2880**, MoE с 32 экспертами, квантование MXFP4). Данные и сплиты те
же: `data/main.jsonl` (6000), пилот — первые 50 строк того же входа.

### 15.1. Режим: harmony-промпт с предзаполненным каналом `final`

Штатные способы отключить reasoning у gpt-oss в инструменте не работают:

- `--no-thinking` шаблон gpt-oss игнорирует — в промпте всё равно `Reasoning: medium`;
- `--reasoning on|off` и `--reasoning-budget` действуют через сэмплер, а генерация в
  инструменте — сырой greedy-argmax без сэмплера (`hidden-states.cpp:846-884`);
- `--chat-template-kwargs` инструмент не передаёт.

Обход без правок кода — предзаполнить канал `final`. В JSONL заменяется только
`question` (остальные ключи строки сохраняются), запуск идёт с `--no-template`:

```
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2026-10-21

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|><вопрос>
Answer with just the answer, no explanation.<|end|><|start|>assistant<|channel|>final<|message|>
```

Вход собирает `pipeline/make_gptoss_prompt.py` (проверен: побайтово воспроизводит
использованный файл `gptoss20b_final6000_q_v2.jsonl`; инструкция в промпте ровно одна —
это же проверяется в скрипте).

Точка A при таком промпте — последний токен нашего сообщения (`<|message|>` канала
`final`), а не `<|start|>assistant`. Это осознанное отклонение: без него 64 токена
уходят в канал `analysis` и всё обрезается (0/8 при 64 токенах, 4/8 при 128).

### 15.2. Пилот и полный прогон

Флаги: `--generate --max-new-tokens 64 --max-len 1024 --layers all --no-thinking
--no-template -ngl 11 --batch-rows 8`.

| Что | Пилот 50 | Полный (`out/main_gptoss20b_v2`) |
|---|---|---|
| примеров | 50 | 6000 |
| ok / skipped | 50 / 0 | 6000 / 0 |
| `truncated` | 0 | **27 (0.45 %)** |
| время | 46.1 s | **10733.5 s** |
| скорость | 1.08 ex/s | **0.56 ex/s** |
| retokenize mismatch | 0/50 | 21/6000 (0.3 %) |
| массивы | `[50, 24, 2880]` | `[6000, 24, 2880]`, по 829.4 МБ |
| `n_tokens` min/med/max | — | 87 / 101 / 210 |

Скорость полного прогона **не показатель батчинга**: первые ~2100 строк шли на
1.06 ex/s (столько же, сколько пилот), затем темп упал до 0.4–0.6 ex/s. Причина —
память машины, а не вход: свободной RAM оставалось 0.7–1 ГБ при 16 ГБ всего (процесс
занимает ~12 ГБ, из них 11 слоёв ≈ 4.6 ГБ на GPU), GPU при этом 99 %, 41–44 Вт, 53 °C,
без троттлинга. Тот же набор в первом прогоне (с дефектным входом, §15.2.1) прошёл за
5596 s (1.07 ex/s) — там память ещё не была зажата.

Пустых ответов и пустых `logprobs` нет; дубль `id` `bb_956` в `main.jsonl` (две разные
формулировки вопроса) обработан дважды, как и в прогонах Qwen/Gemma.

#### 15.2.1. Первый прогон и почему он не пошёл в отчёты

Первый полный прогон (6000/6000, 5596 s, 1.07 ex/s, `truncated` 23, strict 0.501)
сделан на входе, где инструкция `Answer with just the answer, no explanation.` стояла
в промпте **дважды**: она уже есть в `question` из `main.jsonl` (`prepare_data.py` её
добавляет), а сборщик harmony-промпта добавил ещё раз. Промпт не совпадал с пилотом,
поэтому числа промаркированы как недействительные. Результат сохранён в
`out/main_gptoss20b_dupinstr` (в отчёты не входит), а прогон повторён на исправленном
входе. Для справки: качество от этого почти не изменилось (strict 0.501 против 0.494,
проба C/16 test AUROC 0.919 против 0.925), но формально это другой промпт.

### 15.3. Разметка

| правило | единиц / 5973 | доля |
|---|---|---|
| strict | 2948 | 0.494 |
| relaxed | 2973 | 0.498 |
| filtered_a | 2910 | 0.487 |
| **filtered_b (итоговое)** | **2928** | **0.490** |

Сплиты (strict): train 1795/3588 = 0.500, val 569/1193 = 0.477, test 584/1192 = 0.490.
Без метки 27 строк — ровно все `truncated`; пустых и вырожденных, как у Gemma, нет.
Фильтр `a` меняет 38 меток, `b` — 20; между собой расходятся на 18 строках.

Доля правильных у gpt-oss-20b в **2.7 раза выше**, чем у Qwen3.5-0.8B (0.143) и
Gemma 4 E2B (0.182): 20B-модель отвечает на TriviaQA заметно лучше. Метка у каждой
модели своя (правильность её собственного ответа), поэтому AUPRC между моделями
напрямую не сравнивать: при доле положительных 0.49 AUPRC высок по построению.

### 15.4. Baseline

| признак | split | AUROC | AUPRC | Brier | ECE |
|---|---|---|---|---|---|
| logprob | val | 0.835 | 0.829 | — | — |
| logprob | test | 0.828 | 0.828 | — | — |
| entropy | val | 0.850 | 0.844 | — | — |
| entropy | test | 0.841 | 0.841 | — | — |
| logprob, калибр. | test | 0.828 | 0.828 | 0.174 | 0.057 |
| entropy, калибр. | test | 0.841 | 0.841 | 0.165 | 0.055 |
| сырая `exp(lp0)` | test | 0.828 | 0.828 | 0.175 | 0.064 |

Baseline сильнее, чем у Qwen (0.739/0.767) и Gemma (0.705/0.712), но и доля правильных
втрое выше — сравнение AUROC между моделями с разной базовой долей ограничено.

### 15.5. Проба

Сетка `λ` та же: средние val AUROC 100 → 0.7852, 300 → 0.8005, 1000 → 0.8117,
**3000 → 0.8147**. Выбрано `λ = 3000`, **точка C, слой 16**.

| split | AUROC [95 % CI] | AUPRC [95 % CI] | Brier | ECE |
|---|---|---|---|---|
| val | 0.9157 [0.900, 0.930] | 0.9057 | 0.1177 | 0.0418 |
| **test** | **0.9248 [0.909, 0.939]** | **0.9197 [0.894, 0.937]** | **0.1109** | **0.0531** |

Platt-калибровка: test Brier 0.1087, ECE 0.0312.

**Кривая val AUROC по слоям (λ = 3000):**

| слой | A | C |
|---|---|---|
| 0 | 0.671 | 0.669 |
| 1 | 0.687 | 0.705 |
| 2 | 0.694 | 0.729 |
| 3 | 0.707 | 0.744 |
| 4 | 0.712 | 0.766 |
| 5 | 0.734 | 0.782 |
| 6 | 0.737 | 0.797 |
| 7 | 0.740 | 0.818 |
| 8 | 0.759 | 0.843 |
| 9 | 0.766 | 0.862 |
| 10 | 0.782 | 0.877 |
| 11 | 0.785 | 0.882 |
| 12 | 0.790 | 0.899 |
| 13 | 0.793 | 0.902 |
| 14 | 0.810 | 0.914 |
| 15 | 0.837 | 0.914 |
| **16** | 0.843 | **0.916** |
| 17 | 0.849 | 0.909 |
| 18 | 0.852 | 0.906 |
| 19 | 0.857 | 0.901 |
| 20 | 0.857 | 0.899 |
| 21 | 0.853 | 0.893 |
| 22 | 0.851 | 0.888 |
| 23 | 0.843 | 0.882 |

Контроли: перемешанные метки — test AUROC 0.494 ± 0.074; слой 0 — A 0.632, C 0.684.
Разрыв val→test +0.009 (тест чуть выше val, как у Gemma).

Отличие от двух малых моделей: лучший слой — 16 из 24 (**67 % глубины**) против ~54 %
у Qwen (13/24) и Gemma (19/35). Сетку `λ`, точки и слои под это не подгоняли.

### 15.6. Сравнение трёх моделей

| | Qwen3.5-0.8B | Gemma 4 E2B | gpt-oss-20b |
|---|---|---|---|
| слоёв × D | 24 × 1024 | 35 × 1536 | 24 × 2880 |
| правильных (filtered_b) | 0.143 | 0.182 | 0.490 |
| baseline logprob, test AUROC | 0.739 | 0.705 | 0.828 |
| baseline entropy, test AUROC | 0.767 | 0.712 | 0.841 |
| проба: λ | 1000 | 1000 | 3000 |
| проба: точка, слой | C, 13 | C, 19 | C, 16 |
| проба: val AUROC | 0.890 | 0.908 | 0.9157 |
| **проба: test AUROC** | **0.857** [0.825, 0.885] | **0.935** [0.918, 0.950] | **0.9248** [0.909, 0.939] |
| проба: test AUPRC | 0.593 | 0.744 | 0.9197 (доля + 0.49) |
| контроль: слой 0 (C), test | 0.718 | 0.780 | 0.684 |
| контроль: перемешанные метки | 0.490 ± 0.052 | 0.521 ± 0.056 | 0.494 ± 0.074 |

Выводы:

- скрытые состояния gpt-oss отделяют верный ответ от неверного почти так же, как
  у Gemma 4 E2B (0.925 против 0.935), и лучше, чем у Qwen3.5-0.8B (0.857);
- у gpt-oss контроль «слой 0» ниже (0.684), чем у Gemma (0.780), то есть прирост от
  глубоких слоёв больше;
- AUPRC 0.92 у gpt-oss выше «потолочного» AUPRC малых моделей просто потому, что
  правильных ответов половина; сравнение AUPRC между моделями здесь неинформативно.

### 15.7. Батчирование строк: что дало

Замеры на 200 строках того же типа (`gptoss20b_final200_q.jsonl`, `-ngl 11`):

| режим | `--batch-rows 1` | 4 | 8 | ускорение b8 |
|---|---|---|---|---|
| только состояния | 229.5 s (0.87 ex/s) | 68.6 s | **43.7 s (4.58 ex/s)** | ×5.2 |
| состояния + генерация | 533 s (0.37 ex/s) | 235.2 s | **193.2 s (1.04 ex/s)** | ×2.8 |

На полном наборе: 5596 s в первом прогоне (1.07 ex/s) при оценке одиночного режима
~16 000 s (0.37 ex/s, 4.4 ч) — ускорение ×2.9. Во втором прогоне темп упал до 0.56 ex/s
из-за нехватки RAM (§15.2), а не из-за батчинга.

Корректность батчинга проверена отдельно (§7 и `AGENTS.md`): при `--batch-rows 1`
результат побайтово совпадает с прежним одиночным; при 8 ответы совпадают текстом в
170/200 случаев, но метки — в 200/200; состояния расходятся на 0.1–0.5 % (cos median
1.00000, min 0.9987); повторный прогон побайтово идентичен. На малых моделях (Qwen,
Gemma) батчинг тоже работает, но выигрыш ~×1.2 — они упираются в накладные расходы.

### 15.8. Артефакты

```
python make_gptoss_prompt.py --in data\main.jsonl --out ..\..\..\gptoss20b_final6000_q_v2.jsonl

llama-hidden-states.exe -m gpt-oss-20b-MXFP4.gguf \
  --batch gptoss20b_final6000_q_v2.jsonl \
  --generate --max-new-tokens 64 --max-len 1024 --layers all --no-thinking --no-template \
  -ngl 11 --batch-rows 8 --out out\main_gptoss20b_v2

python label.py --meta out\main_gptoss20b_v2\meta.jsonl --out out\main_gptoss20b_v2\labels.csv
python baseline.py --meta out\main_gptoss20b_v2\meta.jsonl --labels out\main_gptoss20b_v2\labels.csv \
  --out out\main_gptoss20b_v2\baseline.json --scores-out out\main_gptoss20b_v2\baseline_scores.csv
python probe.py --hidden-a out\main_gptoss20b_v2\hidden_a.npy --hidden-c out\main_gptoss20b_v2\hidden_c.npy \
  --meta out\main_gptoss20b_v2\meta.jsonl --labels out\main_gptoss20b_v2\labels.csv \
  --out out\main_gptoss20b_v2\probe.json --curve-out out\main_gptoss20b_v2\probe_curve.csv \
  --scores-out out\main_gptoss20b_v2\probe_scores.csv
```

Каталоги: `out/main_gptoss20b_v2/` (действительный прогон), `out/main_gptoss20b/`
(неполный: 325/6000, остановлен вручную, массивы предвыделены), `out/main_gptoss20b_dupinstr/`
(полный, но с дефектным промптом; в отчёты не входит).

---

## 16. Кросс-модельные скоры проб (шаг 9)

**Задача.** Проверить, переносится ли проба между моделями. Перенос весов невозможен:
D = 1024 (Qwen), 1536 (Gemma), 2880 (gpt-oss), а проба линейна по вектору состояния.
Проверен более слабый, но корректный вариант: скор одной модели против **метки другой**
на общих строках одного сплита (`pipeline/cross_probe_scores.py`). Метки у моделей свои
(правильность собственного ответа), поэтому «кросс» измеряет перенос сигнала уверенности,
а не направления в пространстве.

**Метод.** Строки `probe_scores.csv` (val+test) склеиваются по `row`; берутся только
строки, которые обе модели отдали в одном сплите и у которых есть непустая метка
`correct_filtered_b`. Родные AUROC пересчитываются на том же подмножестве. Интервалы —
bootstrap по строкам (1000, seed 42). Контроль: AUROC(метка источника, метка цели) —
сколько даёт одно знание исхода второй модели.

**Тест** (общие строки 1186–1189):

| источник скора → метка | родной AUROC источника | родной AUROC цели | кросс AUROC [95 % CI] | метка → метка | согласие меток |
|---|---|---|---|---|---|
| Qwen → Gemma | 0.858 | 0.936 | 0.771 [0.732, 0.807] | 0.722 | 0.849 |
| Gemma → Qwen | 0.936 | 0.858 | 0.753 [0.712, 0.792] | 0.705 | 0.849 |
| gpt-oss → Qwen | 0.925 | 0.857 | 0.801 [0.766, 0.836] | 0.735 | 0.633 |
| Qwen → gpt-oss | 0.857 | 0.925 | 0.679 [0.648, 0.709] | 0.620 | 0.633 |
| gpt-oss → Gemma | 0.925 | 0.935 | 0.832 [0.803, 0.863] | 0.742 | 0.644 |
| Gemma → gpt-oss | 0.935 | 0.925 | 0.690 [0.658, 0.719] | 0.633 | 0.644 |

**Val** (1186 строк) даёт ту же картину: Qwen→Gemma 0.756, Gemma→Qwen 0.779,
gpt-oss→Qwen 0.780, Qwen→gpt-oss 0.668, gpt-oss→Gemma 0.802, Gemma→gpt-oss 0.689;
родные AUROC 0.891/0.909 (Qwen/Gemma), 0.916/0.915 (gpt-oss с каждой из малых).

**Наблюдения.**

- Перенос частичный: кросс AUROC 0.68–0.83 против 0.86–0.94 у родных проб. До 0.5
  («совсем не переносится») не падает — общий сигнал есть.
- Заметная его часть — сложность вопроса: контроль «знать исход второй модели» даёт
  0.62–0.74, согласие меток 63–85 %; кросс-скор выше контроля во всех направлениях
  (+0.03…+0.09).
- Асимметрия связана с базовыми долями: 0.49 правильных у gpt-oss против 0.15–0.16
  у малых; редкое событие («малая модель права») труднее ранжировать.
- Корреляция скоров 0.44–0.50. AUPRC между парами не сравнивается (разные доли).
- Стратификация (§16.1) показывает, что часть сигнала остаётся и при фиксированном
  исходе источника, то есть это не только общая бинарная правильность.

**Артефакты:** `out/cross_qwen08_gemma4e2b_{test,val}.json`,
`out/cross_qwen08_gptoss20b_v2_{test,val}.json`,
`out/cross_gemma4e2b_gptoss20b_v2_{test,val}.json`; скрипт
`pipeline/cross_probe_scores.py`.

### 16.1 Стратификация по метке источника

Чтобы отделить «известно, права ли A» от сигнала в самом скоре, строки делятся по метке
A, и AUROC(скор A, метка B) считается внутри каждой страты (`cross_probe_strata.py`).
Дополнительно логистика «метка A → метка B» сравнивается с «метка A + скор A»; обучение
на одном сплите, проверка на другом.

| направление | overall | внутри A верно | внутри A неверно | метка A → +скор A |
|---|---|---|---|---|
| Qwen → Gemma | 0.771 | 0.712 [0.633, 0.786] | 0.675 [0.617, 0.735] | 0.705 → 0.794 (+0.089) |
| Gemma → Qwen | 0.753 | 0.587 [0.512, 0.672] | 0.614 [0.549, 0.676] | 0.722 → 0.773 (+0.051) |
| gpt-oss → Qwen | 0.801 | 0.689 [0.637, 0.737] | 0.622 [0.490, 0.745] | 0.735 → 0.811 (+0.076) |
| Qwen → gpt-oss | 0.679 | 0.821 [0.742, 0.896] | 0.599 [0.564, 0.636] | 0.620 → 0.692 (+0.072) |
| gpt-oss → Gemma | 0.832 | 0.730 [0.685, 0.773] | 0.671 [0.550, 0.789] | 0.742 → 0.836 (+0.095) |
| Gemma → gpt-oss | 0.690 | 0.747 [0.633, 0.851] | 0.590 [0.555, 0.626] | 0.633 → 0.697 (+0.064) |

Тест, логистика обучена на val; на val (обучение на test) внутри страт 0.607–0.679,
добавка скора +0.061…+0.103.

**Вывод.** Даже при фиксированном исходе A скор остаётся информативным: 0.59–0.82
внутри страт (в основном 0.60–0.73), добавка к логистике +0.05…+0.10. Значит,
выше-случайный кросс — не только «общая бинарная правильность»: скор несёт
градуированную информацию о трудности вопроса. Оговорки: страты с редким классом
(gpt-oss ошибся, малая модель права: 21–22 положительных) дают широкие интервалы;
переносом весов это по-прежнему не является, и отдельного контроля на длину
вопроса/logprob не делалось.

**Артефакты:** `out/cross_strata_<a>_<b>_{test,val}.json` (12 файлов); скрипт
`pipeline/cross_probe_strata.py`.

---

## 17. Четвёртая модель: Nemotron 3 Nano 4B (шаг 10)

Модель: `NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf`, архитектура `nemotron_h` — **гибрид**:
42 слоя, **D = 3136**, чередование Mamba/SSM и attention блоков. Данные и сплиты те же:
`data/main.jsonl` (6000), пилот — `data/pilot.jsonl` (200).

### 17.1. Особенность архитектуры: `l_out` есть только на 17 слоях из 42

У `nemotron_h` скрытое состояние `l_out` эмитится только в `build_ffn_layer`
(`src/models/nemotron-h.cpp:255`). SSM/attention слои отдают наружу лишь
`nemotron_h_block_out`, поэтому `l_out` существует ровно на 17 слоях:

```
1, 3, 5, 8, 10, 13, 15, 18, 20, 22, 25, 27, 29, 33, 37, 39, 41
```

Прежняя версия инструмента требовала захвата **всех** выбранных слоёв
(`st.n_cap_c != L_sel` → `skipped`). При `--layers all` выбиралось 42 слоя,
захватывалось 17, и каждая строка получала `status: skipped`, а массивы оставались
нулевыми (файлы при этом заранее размечены под полный размер, поэтому «выглядели
правильными»). Так были потеряны прогон `out/main_nemotron3nano4b` (5266 строк, 2.7 ч)
и пилот `out/pilot_nemotron3nano4b` (там же повторный запуск в тот же каталог перезаписал
`config.json`). Оба каталога помечены `README-invalid.md` и в анализ не входят.

Что исправлено в инструменте (коммит вместе с этим разделом):

- перед прогоном делается один пробный decode на BOS, собираются реально существующие
  `l_out-N`, и выбранные слои без `l_out` выбрасываются с предупреждением в лог;
  исходный список сохраняется в `config.json` как `layers_requested`;
- `write_skipped` больше не затирает сгенерированный ответ (`generated`, `truncated`,
  `n_tokens`, `pos_a`, `pos_c`, `logprobs`, `entropies` пишутся как есть);
- при неполном захвате в лог пишется, каких именно слоёв не хватило.

Побочно: прежние «результаты» Nemotron (AUROC 0.500, accuracy 1.000) были получены
скриптами-обходами на нулевых массивах и удалены; для проверки каталога добавлен
`pipeline/verify_run.py`.

### 17.2. Прогон

Флаги: `--generate --max-new-tokens 64 --max-len 1024 --layers all -ngl 99`
(batch-rows 1, `--out out/main_nemotron3nano4b_v2`).

| Что | Пилот 200 | Полный (`out/main_nemotron3nano4b_v2`) |
|---|---|---|
| примеров | 200 | 6000 |
| ok / skipped | 200 / 0 | **6000 / 0** |
| `truncated` | 0 | **5 (0.08 %)** |
| время | 44.7 s | **1320.6 s** (22 мин) |
| скорость | 4.48 ex/s | **4.54 ex/s** |
| retokenize mismatch | 0/200 | 19/6000 (0.3 %) |
| массивы | `[200, 17, 3136]` | `[6000, 17, 3136]`, по 610 МБ |
| слоёв в `config.json` | 17 (`layers_requested` 42) | 17 (`layers_requested` 42) |

По времени это самый медленный из трёх прогонов малых моделей, но разрыв мал:
1320.6 s против 1258.1 s у Gemma 4 E2B (4.77 ex/s) и 1287.1 s у Qwen3.5-0.8B
(4.66 ex/s) — 3–5 %. То есть 17 слоёв вместо 24–35 и короткие ответы (медиана 3 токена)
не дают выигрыша: прогон упирается не в число захватываемых слоёв, а в сам проход по
42-слойной сети на каждый токен. gpt-oss для сравнения — 10733.5 s. Батчинг здесь не
помогает: на 50 строках `--batch-rows 1` и `4` дают одинаковые 10.4 с (доминирует
генерация), поэтому полный прогон сделан с `batch-rows 1`.

Ответы короткие: медиана 3 токена, среднее 3.6, максимум 64 (все 5 `truncated`).
Пустых ответов нет, вырожденных (`}`, `?`, `<|channel>thought`, как у Gemma) тоже нет.
Дубль `id` `bb_956` из `main.jsonl` обработан дважды, как и в остальных прогонах.
`verify_run.py --full` по каталогу: 0 failed (единственное предупреждение — про
выброшенные 25 слоёв без `l_out`).

### 17.3. Разметка

| правило | единиц / 5995 | доля |
|---|---|---|
| strict | 1767 | 0.295 |
| relaxed | 1777 | 0.296 |
| filtered_a | 1693 | 0.282 |
| **filtered_b (итоговое)** | **1708** | **0.285** |

Сплиты (strict): train 1097/3598 = 0.305, val 327/1197 = 0.273, test 343/1200 = 0.286.
Без метки 5 строк — ровно все `truncated`. Strict и relaxed расходятся на 10 строках
(relaxed добавляет 10, ни одной не убирает). Фильтр `a` снимает 74 верных ответа, у
которых совпавшая фраза встречается в самом вопросе; фильтр `b` — 59 из них (у 15 фраза
совпадает с эталоном `gold_main`, и такой ответ не снимается).

Доля правильных 0.285 — между малыми моделями (Qwen 0.143, Gemma 0.182) и gpt-oss (0.490):
4B-модель отвечает на TriviaQA точнее 0.8B/2B, но заметно хуже 20B. Метка у каждой модели
своя (правильность её собственного ответа), поэтому AUPRC между моделями несопоставим:
при доле положительных 0.285 он ниже по построению, чем при 0.49.

### 17.4. Baseline

| признак | split | AUROC | AUPRC | Brier | ECE |
|---|---|---|---|---|---|
| logprob | val | 0.783 | 0.578 | — | — |
| logprob | test | 0.802 | 0.613 | — | — |
| entropy | val | 0.806 | 0.608 | — | — |
| entropy | test | 0.816 | 0.635 | — | — |
| logprob, калибр. | test | 0.802 | 0.613 | 0.153 | 0.034 |
| entropy, калибр. | test | 0.816 | 0.635 | 0.146 | 0.020 |
| сырая `exp(lp0)` | test | 0.802 | 0.613 | 0.170 | 0.124 |

Baseline сильнее, чем у Qwen (0.739/0.767) и Gemma (0.705/0.712), но слабее gpt-oss
(0.828/0.841) — как и доля правильных. Сырая вероятность первого токена плохо
калибрована (ECE 0.124), калибровка на val это исправляет (0.020–0.034); ранжирование
(ECE) от монотонного преобразования не зависит.

### 17.5. Проба

Сетка `λ` та же: средние val AUROC 100 → 0.7898, 300 → 0.8053, 1000 → 0.8155,
**3000 → 0.8177**. Выбрано `λ = 3000`, **точка C, слой 20** (8-й индекс в списке
17 доступных слоёв).

| split | AUROC [95 % CI] | AUPRC [95 % CI] | Brier | ECE |
|---|---|---|---|---|
| val | 0.9059 [0.887, 0.924] | 0.7930 [0.750, 0.832] | 0.1052 | 0.0469 |
| **test** | **0.8819 [0.860, 0.901]** | **0.7357 [0.685, 0.782]** | **0.1198** | **0.0412** |

Platt-калибровка почти не меняет: test Brier 0.1199, ECE 0.0392.

**Кривая val AUROC по слоям (λ = 3000):**

| слой | A | C |
|---|---|---|
| 1 | 0.681 | 0.709 |
| 3 | 0.704 | 0.748 |
| 5 | 0.745 | 0.772 |
| 8 | 0.740 | 0.803 |
| 10 | 0.767 | 0.825 |
| 13 | 0.777 | 0.851 |
| 15 | 0.788 | 0.881 |
| 18 | 0.803 | 0.896 |
| **20** | 0.833 | **0.906** |
| 22 | 0.835 | 0.903 |
| 25 | 0.845 | 0.900 |
| 27 | 0.838 | 0.874 |
| 29 | 0.828 | 0.857 |
| 33 | 0.811 | 0.850 |
| 37 | 0.799 | 0.830 |
| 39 | 0.838 | 0.852 |
| 41 | 0.852 | 0.860 |

Плато 18–25 (0.896–0.906) — то же поведение, что у остальных моделей: сигнал растёт до
середины сети и выходит на плато, затем медленно падает. Точка C выше A везде; максимум
различия — на слоях 15–18 (0.09), на первых слоях 1–5 оно 0.03–0.04, а на последних
(33–41) падает до 0.01–0.04, то есть к концу сети A и C сближаются.

Контроли: перемешанные метки — test AUROC 0.481 ± 0.035 (5 перестановок); слой 0 —
A 0.649, C 0.650 на тесте. Разрыв val→test **−0.024** (тест ниже val), в пределах
обычного 0.04; обратной аномалии, как у Gemma (+0.027), нет.

Лучший слой — 20 из 42 (**49 % глубины**), чуть мельче, чем у Qwen (13/24 = 57 %),
Gemma (19/35 = 56 %) и gpt-oss (16/24 = 70 %). Сетку `λ`, точки и слои под это не
подгоняли: выбранные слои ограничены архитектурой (17 из 42), и все 17 пройдены одним
проходом по сетке.

### 17.6. Сравнение четырёх моделей

| | Qwen3.5-0.8B | Gemma 4 E2B | gpt-oss-20b | Nemotron 3 Nano 4B |
|---|---|---|---|---|
| слоёв × D | 24 × 1024 | 35 × 1536 | 24 × 2880 | 42 × 3136 (17 с `l_out`) |
| правильных (filtered_b) | 0.143 | 0.182 | 0.490 | 0.285 |
| baseline logprob, test AUROC | 0.739 | 0.705 | 0.828 | 0.802 |
| baseline entropy, test AUROC | 0.767 | 0.712 | 0.841 | 0.816 |
| проба: λ | 1000 | 1000 | 3000 | 3000 |
| проба: точка, слой | C, 13 | C, 19 | C, 16 | C, 20 |
| проба: глубина слоя | 57 % | 56 % | 70 % | 49 % |
| проба: val AUROC | 0.890 | 0.908 | 0.9157 | 0.9059 |
| **проба: test AUROC** | **0.857** [0.825, 0.885] | **0.935** [0.918, 0.950] | **0.9248** [0.909, 0.939] | **0.8819** [0.860, 0.901] |
| проба: test AUPRC | 0.593 | 0.744 | 0.9197 | 0.7357 |
| контроль: слой 0 (C), test | 0.718 | 0.780 | 0.684 | 0.650 |
| контроль: перемешанные метки | 0.490 ± 0.052 | 0.521 ± 0.056 | 0.494 ± 0.074 | 0.481 ± 0.035 |

Парные bootstrap-сравнения на тесте (`compare_probes.py`, ресэмплинг по вопросам, у
каждой модели своя метка):

| пара | ΔAUROC [95 % CI] | ΔAUPRC [95 % CI] | доля выборок, где Nemotron выше |
|---|---|---|---|
| Nemotron vs Qwen | **+0.0255** [−0.005, +0.060] | **+0.148** [+0.079, +0.218] | 0.949 |
| Nemotron vs Gemma | **−0.0536** [−0.079, −0.028] | −0.009 [−0.079, +0.056] | 0.000 |
| Nemotron vs gpt-oss | **−0.0427** [−0.067, −0.021] | −0.183 [−0.230, −0.138] | 0.000 |

Выводы:

- Nemotron 4B по пробам **лучше Qwen3.5-0.8B** (+0.026 AUROC, AUPRC +0.15 из-за вдвое
  большей доли правильных; интервал ΔAUROC чуть задевает ноль), **хуже Gemma 4 E2B**
  (−0.054, значимо) и **хуже gpt-oss-20b** (−0.043 по AUROC и −0.18 по AUPRC).
- По AUPRC с Gemma разницы нет (−0.009, интервал широкий) — но AUPRC у моделей с разной
  базовой долей (0.285 против 0.182) сравнивать нельзя; это совпадение, а не «равенство».
- Разрыв «проба минус baseline» у Nemotron самый маленький из четырёх:
  0.8819 против 0.802 (logprob), то есть **+0.08 AUROC** сверх первого токена; у Qwen
  +0.12, у gpt-oss +0.10, у Gemma +0.23. Причина не исследовалась; вероятная — у 4B
  первый токен уже неплохо ранжирует вопросы (baseline 0.802).
- Контроль «слой 0» у Nemotron самый низкий (0.650), значит прирост от глубоких слоёв
  относительно начала сети больше, чем у остальных: 0.65 → 0.88.

### 17.7. Кросс-модельные скоры с участием Nemotron

Тот же метод, что в §16 (`cross_probe_scores.py`): скор одной модели против метки
другой на общих строках теста. Здесь `b` — Nemotron.

| направление | родной AUROC источника | родной AUROC цели | кросс AUROC [95 % CI] | метка → метка | согласие меток |
|---|---|---|---|---|---|
| Nemotron → Qwen | 0.883 | 0.857 | 0.794 [0.755, 0.829] | 0.775 | 0.799 |
| Qwen → Nemotron | 0.857 | 0.883 | 0.732 [0.697, 0.766] | 0.676 | 0.799 |
| Nemotron → Gemma | 0.881 | 0.935 | 0.812 [0.776, 0.843] | 0.763 | 0.797 |
| Gemma → Nemotron | 0.935 | 0.881 | 0.725 [0.690, 0.758] | 0.682 | 0.797 |
| Nemotron → gpt-oss | 0.882 | 0.925 | 0.744 [0.715, 0.773] | 0.711 | 0.719 |
| gpt-oss → Nemotron | 0.925 | 0.882 | 0.815 [0.786, 0.841] | 0.766 | 0.719 |

Картина как в §16: перенос частичный (0.72–0.82 против родных 0.86–0.94), везде выше
контроля «знать исход второй модели» на +0.02…+0.06, корреляция скоров 0.52–0.58
(у Nemotron с Gemma 0.52, с gpt-oss 0.58 — самая высокая из всех пар). Асимметрия
прежняя и объясняется базовыми долями: скор Nemotron (доля 0.285) хорошо ранжирует
метки моделей с редким событием и хуже — метку gpt-oss (0.49).

### 17.8. Артефакты

```
llama-hidden-states.exe -m NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf \
  --batch data\main.jsonl --generate --max-new-tokens 64 --max-len 1024 \
  --layers all -ngl 99 --out out\main_nemotron3nano4b_v2

python label.py --meta out\main_nemotron3nano4b_v2\meta.jsonl --out out\main_nemotron3nano4b_v2\labels.csv
python verify_run.py --dir out\main_nemotron3nano4b_v2 --full
python baseline.py --meta out\main_nemotron3nano4b_v2\meta.jsonl --labels out\main_nemotron3nano4b_v2\labels.csv \
  --out out\main_nemotron3nano4b_v2\baseline.json --scores-out out\main_nemotron3nano4b_v2\baseline_scores.csv
python probe.py --hidden-a out\main_nemotron3nano4b_v2\hidden_a.npy --hidden-c out\main_nemotron3nano4b_v2\hidden_c.npy \
  --meta out\main_nemotron3nano4b_v2\meta.jsonl --labels out\main_nemotron3nano4b_v2\labels.csv \
  --out out\main_nemotron3nano4b_v2\probe.json --curve-out out\main_nemotron3nano4b_v2\probe_curve.csv \
  --scores-out out\main_nemotron3nano4b_v2\probe_scores.csv
```

Каталоги: `out/main_nemotron3nano4b_v2/` — действительный прогон (6000 строк, 17 слоёв,
`verify.json` внутри); `out/main_nemotron3nano4b/` и `out/pilot_nemotron3nano4b/` —
непригодны (`README-invalid.md`), оставлены как свидетельство ошибки.
Кросс-скоры: `out/cross_{qwen08,gemma4e2b,gptoss20b_v2}_nemotron_{test,val}.json`
(6 файлов). Веса пробы: `probe_model.npz`.

## 18. Эффект кванта: Qwen3.5-0.8B BF16 vs Q4_K_M (шаг 11)

Серия сравнений размеров Qwen 3.5 будет собрана на весах **Q4_K_M от unsloth** (BF16
для 4B+ не влезает в 6 ГБ VRAM). Поэтому нужен якорь: тот же инструмент, тот же вход и
тот же протокол пробы на модели, у которой есть и BF16-, и Q4-прогон. Такой моделью
взят Qwen3.5-0.8B: BF16-эталон — §11–§13 (`out/main_qwen08`), рядом сделан полный
прогон Q4_K_M (`out/main_qwen08_q4`, 532 517 120 байт, unsloth).

Пилот 200 строк до полного прогона вёлся отдельно (`out/pilot_qwen08_q4`,
`out/pilot_qwen08_iq2`). IQ2_XXS как кандидат отброшен: 196/200 `truncated`, генерация
уходит в `<think>` и обрывается на 64 токенах — метки непригодны, в сравнения не идёт.

### 18.1. Прогон

```
llama-hidden-states.exe -m Qwen3.5-0.8B-Q4_K_M.gguf \
  --batch data\main.jsonl --generate --max-new-tokens 64 --max-len 1024 \
  --layers all -ngl 99 --out out\main_qwen08_q4
```

| | BF16 (`main_qwen08`) | Q4_K_M (`main_qwen08_q4`) |
|---|---|---|
| итог | 6000/6000 ok, 0 skipped | 6000/6000 ok, 0 skipped |
| `truncated` | 23 | **22** |
| `retokenize mismatch` | 29/6000 (0.5 %) | 38/6000 (0.6 %) |
| время | 1287.1 s (4.66 ex/s) | **731.8 s (8.20 ex/s)** |
| массивы | `[6000, 24, 1024]` f16 | `[6000, 24, 1024]` f16 |
| `config.json` | template true, thinking false, 24 слоя | то же |

`verify_run.py --full` по Q4-каталогу: **0 failed, 0 warnings** (`verify.json`), дубль
`bb_956` распознан как дубль входного файла, NaN/Inf нет. Квант дал **×1.76 по
скорости** и ровно те же формы/слои — на подготовку данных и захват состояний
квантование не влияет.

### 18.2. Разметка

| | BF16 | Q4_K_M |
|---|---|---|
| strict | 0.168 (1004/5977) | 0.143 (854/5978) |
| relaxed | 0.172 (1027/5977) | 0.147 (881/5978) |
| `correct_filtered_a` | 0.142 | 0.117 |
| `correct_filtered_b` | 0.143 (856/5977) | **0.118** (707/5978) |
| null | 23 (все `truncated`) | 22 (все `truncated`) |
| strict ≠ relaxed | — | 27 строк |
| по сплитам (strict) | — | train 0.144, val 0.124, test 0.157 |

Q4 ниже по всем колонкам: strict 0.143 против 0.168, `correct_filtered_b` 0.118 против
0.143 (в test 150 положительных против 178). То есть Q4 чаще даёт и неверный, и
формально верный, но «пустой/вырожденный» ответ, который отсекает фильтр b. Строки не
удалялись: они получают пустую метку и не входят в оценку.

### 18.3. Baseline

| | val AUROC | test AUROC | test AUPRC | ECE (калибр.) |
|---|---|---|---|---|
| logprob | 0.724 | 0.680 [0.634, 0.722] | 0.241 | 0.031 |
| entropy | 0.742 | 0.717 [0.668, 0.758] | 0.270 | 0.026 |

Сырая `exp(lp0)` плохо калибрована (ECE 0.087 — первый токен почти всегда служебный),
после Platt-калибровки на val — 0.031/0.026. Для сравнения у BF16 test logprob 0.739,
entropy 0.767: Q4-прогон ранжирует чуть хуже, но AUPRC сравнивать нельзя — базовая доля
у него 0.125 против 0.149.

### 18.4. Проба

Протокол не менялся: сетка λ `{100, 300, 1000, 3000}`, точки A/C, все 24 слоя, выбор
пары (точка, слой) по max val AUROC, тест считается один раз.

Средний val AUROC по сетке λ: 0.7987 / 0.8119 / **0.8144** / 0.8053 → λ = 1000
(у BF16 тоже 1000).

| | BF16 | Q4_K_M |
|---|---|---|
| выбранная пара | C, слой 13 (57 % глубины) | **C, слой 13** |
| val AUROC / AUPRC | 0.8895 / 0.6345 | 0.8838 [0.850, 0.912] / 0.5787 |
| test AUROC | 0.8570 [0.8247, 0.8851] | **0.8709** [0.840, 0.900] |
| test AUPRC | 0.5931 | 0.5477 |
| test Brier / ECE | 0.0892 / 0.0189 | 0.0792 / 0.0227 |
| контроль: перемешанные метки | 0.490 ± 0.052 | 0.503 ± 0.064 |
| контроль: слой 0 (C), test | 0.718 | 0.716 |
| контроль: слой 0 (A), test | — | 0.684 |

Выбор (точка C, слой 13, λ = 1000) **совпал с BF16 полностью**, контроли в норме,
разрыв val→test отрицательный (−0.013), то есть тест не «подрос» — аномалий нет.

### 18.5. Сравнение с BF16 и перенос весов

Парный bootstrap по вопросам (`compare_probes.py`):

| сравнение | ΔAUROC [95 % CI] | ΔAUPRC [95 % CI] |
|---|---|---|
| test, BF16 → Q4 | **+0.0138** [−0.0125, +0.0441] | −0.0454 [−0.108, +0.017] |
| val, BF16 → Q4 | −0.0060 [−0.038, +0.0242] | −0.0559 [−0.129, +0.020] |

ΔAUROC в пределах нуля: квант качество пробы не меняет. ΔAUPRC отрицательный только
потому, что у Q4 ниже базовая доля (0.125 против 0.149 в test) — AUPRC между квантами
и между размерами сравнивать нельзя.

Перенос обученных весов (`apply_probe_model.py`, новый скрипт):

| направление | test AUROC | test AUPRC | val AUROC |
|---|---|---|---|
| BF16-проба → состояния Q4 | 0.8677 | 0.5595 | 0.8785 |
| Q4-проба → состояния BF16 | 0.8634 | 0.5838 | 0.8920 |
| родная проба Q4 | 0.8709 | 0.5477 | 0.8838 |

Перенос даёт то же качество, что и обучение заново (0.8677 против 0.8709): направление
разделяющей гиперплоскости между квантами устойчиво, пересчитывать `mu/sd` на новых
состояниях почти не требуется. Практический вывод для серии размеров: проба,
снятая на одном кванте, применима к другому без потери.

### 18.6. Что меняет квант: ответы, метки, состояния

- Ответы текстово совпадают у BF16 и Q4 лишь на **1897/6000 (31.6 %)** строк — при
  greedy-декодировании расхождение в один токен уводит всю реплику.
- Метка `correct_filtered_b` при этом совпадает на **5560/5961 (93.3 %)**: BF16 верно и
  Q4 неверно — 275 строк, наоборот — 126. У `correct_strict` согласие 93.0 %
  (283 против 132), то есть квант чаще ломает именно формально-верные ответы.
- Сдвиг представлений (`state_shift_bf16_vs_q4.json`, 5961 общая годная строка):
  - точка A (один и тот же промпт): cosine median **0.975**, min 0.937, относительный
    L2 median 0.225 — это чистый эффект квантования;
  - точка C на строках с **одинаковым ответом** (1897): cosine median **0.9859**,
    p05 0.9747, min 0.915, относительный L2 median 0.168;
  - точка C по всем строкам: cosine median 0.926 — разница почти целиком объясняется
    разными текстами ответов, а не порчей векторов.

Итог: квант сдвигает скрытые состояния (cosine 0.97–0.99), но не меняет ни выбранную
точку/слой/λ, ни (в пределах CI) качество пробы. Метки у двух квантов — разные
события (Q4 реже отвечает правильно), поэтому в серии размеров каждая модель
сравнивается по своим меткам, а перенос весов пробы между квантами допустим.

### 18.7. Артефакты

```
python label.py --meta out\main_qwen08_q4\meta.jsonl --out out\main_qwen08_q4\labels.csv
python verify_run.py --dir out\main_qwen08_q4 --full --json out\main_qwen08_q4\verify.json
python baseline.py --meta out\main_qwen08_q4\meta.jsonl --labels out\main_qwen08_q4\labels.csv \
  --out out\main_qwen08_q4\baseline.json --scores-out out\main_qwen08_q4\baseline_scores.csv
python probe.py --hidden-a out\main_qwen08_q4\hidden_a.npy --hidden-c out\main_qwen08_q4\hidden_c.npy \
  --meta out\main_qwen08_q4\meta.jsonl --labels out\main_qwen08_q4\labels.csv \
  --out out\main_qwen08_q4\probe.json --curve-out out\main_qwen08_q4\probe_curve.csv \
  --scores-out out\main_qwen08_q4\probe_scores.csv
python compare_probes.py --a out\main_qwen08\probe_scores.csv --b out\main_qwen08_q4\probe_scores.csv \
  --split test --out out\compare_qwen08_bf16_vs_q4_test.json
python apply_probe_model.py --model out\main_qwen08\probe_model.npz --dir out\main_qwen08_q4 \
  --split test --out out\main_qwen08_q4\probe_from_bf16_test.json
```

Каталог `out/main_qwen08_q4/`: `probe.json`, `probe_curve.csv`, `probe_scores.csv`,
`probe_model.npz`, `baseline.json`, `baseline_scores.csv`, `labels.csv`, `meta.jsonl`,
`config.json`, `verify.json`, `probe_from_bf16_{val,test}.json`,
`probe_from_q4_on_bf16_{val,test}.json`, `state_shift_bf16_vs_q4.json`.
Сравнения: `out/compare_qwen08_bf16_vs_q4_{test,val}.json`. Пилоты:
`out/pilot_qwen08_q4`, `out/pilot_qwen08_iq2` (IQ2 — только как свидетельство отказа
генерации).

## 19. Серия размеров Qwen 3.5: 0.8B, 2B, 4B в Q4_K_M (шаг 12)

Якорь кванта (§18) показал, что Q4_K_M не меняет ни выбор пробы, ни качество в пределах
CI, поэтому серия размеров идёт на Q4_K_M от unsloth: BF16 для 4B+ не влезает в 6 ГБ
VRAM. Прогнаны Qwen3.5-2B (24 слоя, D = 2048) и Qwen3.5-4B (32 слоя, D = 2560). Вход,
флаги и протокол пробы — те же, что у якоря; менялся только файл модели. 9B пока не
скачан.

### 19.1. Прогоны

| | 0.8B (`main_qwen08_q4`) | 2B (`main_qwen35_2b_q4`) | 4B (`main_qwen35_4b_q4`) |
|---|---|---|---|
| слоёв × D | 24 × 1024 | 24 × 2048 | 32 × 2560 |
| размер GGUF | 532 МБ | 1.19 ГБ | 2.55 ГБ |
| пилот 200 | — | 28.7 s, 6.96 ex/s, `truncated` 1 | 44.2 s, 4.53 ex/s, `truncated` 1 |
| итог | 6000/6000 ok, 0 skipped | 6000/6000 ok, 0 skipped | 6000/6000 ok, 0 skipped |
| `truncated` | 22 (0.4 %) | 32 (0.5 %) | 20 (0.3 %) |
| `retokenize mismatch` | 38 (0.6 %) | 23 (0.4 %) | 13 (0.2 %) |
| время | 731.8 s (8.20 ex/s) | 866.0 s (6.93 ex/s) | 1483.5 s (4.04 ex/s) |
| массивы | `[6000, 24, 1024]` | `[6000, 24, 2048]` | `[6000, 32, 2560]` |

Пилоты (`out/pilot_qwen35_2b_q4`, `out/pilot_qwen35_4b_q4`) дали 200/200 ok, по одному
`truncated`, `retokenize mismatch` 0/200 и ожидаемые формы — время полного прогона
предсказано ими верно (2B: 28.7 × 30 = 861 s против 866 s; 4B: 44.2 × 30 = 1326 s
против 1483.5 s). `verify_run.py --full` по обоим каталогам: **0 failed, 0 warnings**.

Оговорка по времени 4B: прогон шёл параллельно с CPU-пробой 2B, поэтому 4.04 ex/s —
оценка снизу (в пилоте 4.53 ex/s, в одиночном режиме ожидаемо примерно столько же).

### 19.2. Разметка

| | 0.8B Q4 | 2B Q4 | 4B Q4 |
|---|---|---|---|
| strict | 0.143 (854/5978) | 0.265 (1582/5967) | 0.456 (2727/5980) |
| relaxed | 0.147 (881) | 0.268 (1600) | 0.460 (2749) |
| `filtered_a` | 0.117 (700) | 0.251 (1495) | 0.451 (2695) |
| `filtered_b` | 0.118 (707) | 0.253 (1507) | 0.454 (2714) |
| null | 22 (все `truncated`) | 33 (32 `truncated` + 1 вырожденный) | 20 (все `truncated`) |
| strict по сплитам | 0.144 / 0.124 / 0.157 | 0.269 / 0.243 / 0.275 | 0.467 / 0.434 / 0.445 |

Доля правильных растёт с размером монотонно и сильно: 0.118 → 0.253 → 0.454 по
`filtered_b`. Пустых ответов нет ни у одной модели. У 2B нашёлся один вырожденный ответ
(`#`, `dpql_1540`, row 2083, 41 токен) — он получил пустую метку, строка не удалена;
у 4B вырожденных нет. `strict ≠ relaxed` у 2B на 18 строках, у 4B — на 22 (у 0.8B Q4 — 27).

Пилот дал ту же картину, что и полный прогон (2B: strict 0.246 на 199 строках, 4B:
0.457), — на большем наборе разметка не «поехала».

### 19.3. Baseline

| | 0.8B Q4 | 2B Q4 | 4B Q4 |
|---|---|---|---|
| logprob, val | 0.724 | 0.779 | 0.866 |
| logprob, test | 0.680 [0.634, 0.722] | 0.783 [0.754, 0.813] | 0.870 [0.848, 0.890] |
| entropy, test | 0.717 [0.668, 0.758] | 0.794 [0.765, 0.822] | 0.878 [0.856, 0.897] |
| test AUPRC (logprob / entropy) | 0.241 / 0.270 | 0.583 / 0.584 | 0.842 / 0.848 |
| ECE после калибровки | 0.031 / 0.026 | 0.034 / 0.028 | 0.060 / 0.033 |
| сырая `exp(lp0)`, ECE | 0.087 | 0.087 | 0.054 |

Сырая вероятность первого токена плохо калибрована у всех (ECE 0.054–0.087: первый токен
почти всегда служебный), после Platt на val — 0.03–0.06. Главное — насколько baseline
информативен: 0.680 → 0.783 → 0.870 по logprob. Это и определяет, много ли добавляет
проба (§19.6).

### 19.4. Проба

Протокол не менялся: сетка λ `{100, 300, 1000, 3000}`, точки A/C, все слои, пара
(точка, слой) по max val AUROC, тест считается один раз.

| | 0.8B Q4 | 2B Q4 | 4B Q4 |
|---|---|---|---|
| средний val AUROC по λ (100/300/1000/3000) | 0.799 / 0.812 / **0.814** / 0.805 | 0.791 / 0.807 / 0.817 / **0.818** | 0.810 / 0.827 / 0.840 / **0.845** |
| λ | 1000 | **3000** | **3000** |
| пара | C, 13 | C, **12** | C, **15** |
| глубина слоя | 57 % | 52 % | 48 % |
| val AUROC / AUPRC | 0.8838 / 0.5787 | 0.8948 [0.873, 0.915] / 0.7638 | 0.9218 [0.906, 0.936] / 0.9048 |
| test AUROC | 0.8709 [0.840, 0.900] | **0.8855** [0.865, 0.906] | **0.9180** [0.902, 0.933] |
| test AUPRC | 0.5477 | 0.7587 [0.711, 0.801] | 0.9025 [0.881, 0.923] |
| test Brier / ECE | 0.0792 / 0.0227 | 0.1143 / 0.0447 | 0.1142 / 0.0570 |
| контроль: перемешанные метки (test) | 0.503 ± 0.064 | 0.525 ± 0.016 | 0.522 ± 0.038 |
| контроль: слой 0, C (test) | 0.716 | 0.644 | 0.659 |
| контроль: слой 0, A (test) | 0.684 | 0.656 | 0.672 |

Контроли в норме у обеих моделей: перемешанные метки ≈ 0.5, слой 0 на 0.20–0.27 ниже
победителя. Разрыв val→test отрицательный (2B: −0.009, 4B: −0.004) — тест не «подрос»,
аномалий нет.

Точная цифра слоя у новых размеров неустойчива: у 2B слои 9–14 лежат в 0.885–0.895
(12 против 13 — 0.002), у 4B слои 13–18 в 0.916–0.922 (15 против 16 — 0.002). Устойчива
не конкретная цифра, а **глубина ~50 %**: 12 из 24 и 15 из 32.

Чувствительность к λ тоже низкая: в выбранной паре λ = 1000 даёт у 2B 0.8943 против
0.8948 при 3000, у 4B — 0.9190 против 0.9218, то есть сдвиг λ меняет val на
0.0005–0.003 и на выводы не влияет.

### 19.5. Сравнение размеров

Парный bootstrap по вопросам (`compare_probes.py`, 1000 итераций): у каждой модели своя
метка, поэтому ΔAUROC включает и разную долю правильных ответов.

| сравнение | ΔAUROC test [95 % CI] | доля bootstrap, где B лучше | ΔAUROC val [95 % CI] |
|---|---|---|---|
| 2B vs 0.8B | +0.0154 [−0.019, +0.051] | 0.81 | +0.0116 [−0.024, +0.049] |
| **4B vs 0.8B** | **+0.0482 [+0.018, +0.080]** | **1.00** | +0.0388 [+0.004, +0.077] |
| **4B vs 2B** | **+0.0321 [+0.008, +0.054]** | **1.00** | +0.0271 [+0.003, +0.051] |

Метки различаются сильно: 2B и 0.8B расходятся на 237 из 1186 общих test-строк (20 %),
4B и 0.8B — на 421 из 1194 (35 %), 4B и 2B — на 299 из 1188 (25 %). Базовые доли в test:
0.126 / 0.257 / 0.445, поэтому ΔAUPRC (+0.21 / +0.36 / +0.14) отражает прежде всего рост
доли правильных, а не качество ранжирования; сравнивать его между размерами нельзя.

Итог: **2B статистически не отличается от 0.8B** (интервал задевает ноль, в 19 %
bootstrap-выборок 0.8B выше), **4B достоверно лучше обеих** — интервал не задевает ноль
ни на test, ни на val, а на test во всех 1000 bootstrap-выборках 4B выше.

### 19.6. Что меняется с размером

- **Слой:** ~50 % глубины у всех трёх размеров (13/24, 12/24, 15/32) — как у Gemma
  (19/35, 56 %), gpt-oss (16/24, 70 %) и Nemotron (20/42, 49 %). Точка везде C.
- **Вклад пробы падает по мере роста baseline:**

  | | лучший baseline, test | проба, test | прирост |
  |---|---|---|---|
  | 0.8B Q4 | 0.717 (entropy) | 0.871 | **+0.154** |
  | 2B Q4 | 0.794 (entropy) | 0.886 | **+0.092** |
  | 4B Q4 | 0.878 (entropy) | 0.918 | **+0.040** |

  У 4B скрытые состояния добавляют к первому токену немного: сам baseline уже 0.878.
  Это не дефект пробы, а следствие того, что уверенность первого токена у сильной модели
  лучше связана с правильностью ответа.
- **Базовая доля растёт** (0.118 → 0.253 → 0.454), поэтому AUPRC между размерами не
  сравнивать: для 4B AUPRC 0.90 «высок по построению», как у gpt-oss (0.490).
- **Калибровка пробы хуже** при большей доле: ECE 0.023 → 0.045 → 0.057. Brier при этом
  остаётся вдвое ниже, чем у константного предсказания базовой доли (2B: 0.114 против
  0.190; 4B: 0.114 против 0.247), — проба не «угадывает долю», а ранжирует.
- **Скорость падает** примерно линейно с размером: 8.20 → 6.93 → 4.04 ex/s, полный прогон
  12–25 мин. Серия до 9B на 6 ГБ остаётся доступной.

### 19.7. Артефакты

```
llama-hidden-states.exe -m Qwen3.5-2B-Q4_K_M.gguf \
  --batch data\main.jsonl --generate --max-new-tokens 64 --max-len 1024 \
  --layers all -ngl 99 --out out\main_qwen35_2b_q4

# то же для 4B: --out out\main_qwen35_4b_q4

python label.py --meta out\main_qwen35_2b_q4\meta.jsonl --out out\main_qwen35_2b_q4\labels.csv
python verify_run.py --dir out\main_qwen35_2b_q4 --full --json out\main_qwen35_2b_q4\verify.json
python baseline.py --meta out\main_qwen35_2b_q4\meta.jsonl --labels out\main_qwen35_2b_q4\labels.csv \
  --out out\main_qwen35_2b_q4\baseline.json --scores-out out\main_qwen35_2b_q4\baseline_scores.csv
python probe.py --hidden-a out\main_qwen35_2b_q4\hidden_a.npy \
  --hidden-c out\main_qwen35_2b_q4\hidden_c.npy --meta out\main_qwen35_2b_q4\meta.jsonl \
  --labels out\main_qwen35_2b_q4\labels.csv --out out\main_qwen35_2b_q4\probe.json \
  --curve-out out\main_qwen35_2b_q4\probe_curve.csv \
  --scores-out out\main_qwen35_2b_q4\probe_scores.csv
python compare_probes.py --a out\main_qwen08_q4\probe_scores.csv \
  --b out\main_qwen35_2b_q4\probe_scores.csv --split test \
  --out out\compare_qwen08q4_qwen35_2b_test.json
```

Каталоги `out/main_qwen35_2b_q4/`, `out/main_qwen35_4b_q4/`: `probe.json`,
`probe_curve.csv`, `probe_scores.csv`, `probe_model.npz`, `baseline.json`,
`baseline_scores.csv`, `labels.csv`, `meta.jsonl`, `config.json`, `verify.json`.
Сравнения: `out/compare_qwen08q4_qwen35_{2b,4b}_{test,val}.json`,
`out/compare_qwen35_2b_4b_{test,val}.json` (6 файлов). Пилоты:
`out/pilot_qwen35_2b_q4`, `out/pilot_qwen35_4b_q4`.

### 19.8. Кросс-скоры внутри серии (что даёт скор одной модели на метке другой)

Веса проб между размерами не переносятся: D = 1024 / 2048 / 2560, у пробы
`w` длиной D + 1. `apply_probe_model.py` на такой паре отказывается:
`target D=1024 != probe D=2560`. Переносить можно только **предсказания**:
скор одной модели против метки другой (`cross_probe_scores.py`, метод §16),
на общих строках одного сплита.

**Тест:**

| источник скора → метка | родной AUROC источника | родной AUROC цели | кросс AUROC [95 % CI] | метка → метка | согласие меток | Δ над контролем |
|---|---|---|---|---|---|---|
| 2B → 0.8B | 0.8855 | 0.8701 | 0.7964 [0.760, 0.834] | 0.7687 | 0.800 | +0.028 |
| 4B → 0.8B | 0.9188 | 0.8706 | 0.7945 [0.760, 0.825] | 0.7413 | 0.647 | +0.053 |
| 0.8B → 2B | 0.8701 | 0.8855 | 0.7310 [0.694, 0.766] | 0.6554 | 0.800 | +0.076 |
| 0.8B → 4B | 0.8706 | 0.9188 | 0.7045 [0.675, 0.734] | 0.6073 | 0.647 | +0.097 |
| 4B → 2B | 0.9175 | 0.8854 | 0.8207 [0.791, 0.848] | 0.7910 | 0.748 | +0.030 |
| 2B → 4B | 0.8854 | 0.9175 | 0.7507 [0.723, 0.780] | 0.7247 | 0.748 | +0.026 |

Колонка «метка → метка» — контроль «знать исход другой модели»: AUROC, который даёт
бинарная метка источника как предиктор метки цели (считается в том же JSON как
`cross_label_control`). На val картина та же: кросс AUROC 0.682–0.845 при контроле
0.585–0.835, Δ над контролем +0.001…+0.078 (в направлении 2B → 0.8B — 0.0015, то есть
ничего).

**Как это читать.** Кросс AUROC 0.70–0.82 нельзя сравнивать с 0.5: у моделей одной
серии метки согласуются на 65–80 % строк, и одно знание «ответила ли правильно другая
модель» уже даёт 0.61–0.79. Сверх этого контроля скор добавляет лишь +0.03…+0.10 (на
val местами ≈ 0). Значит, кросс-скор в первую очередь измеряет **общую сложность вопросов
и коррелированность правильности**, а не перенос «направления правильности» между
размерностями. Это согласуется с §16 для разных архитектур (там скор добавлял
+0.03…+0.09 сверх того же контроля) и с ограничением из README §6: перенос весов требует
выученного выравнивания пространств, которое не делалось.

Асимметрия ожидаемая: скор сильной модели лучше ранжирует метку слабой (4B → 0.8B 0.7945),
чем наоборот (0.8B → 4B 0.7045) — редкое событие («0.8B права», доля 0.126) ранжировать
труднее, чем частое (доля 0.445 у 4B). Корреляция скоров между размерами 0.42–0.62 —
то есть скоры и сами по себе похожи, что тоже указывает на общий фактор сложности.

Стратификация по метке источника (§16.1, `cross_probe_strata.py`) для пар внутри серии
**не делалась** — вопрос «остаётся ли сигнал при фиксированном исходе другой модели» для
размеров Qwen 3.5 открыт.

**Артефакты:** 12 файлов `out/cross_{qwen08q4,qwen35_2b,qwen35_4b}_to_..._{test,val}.json`
(6 направлений × 2 сплита); скрипт `pipeline/cross_probe_scores.py`. Перенос весов через
выученное выравнивание пространств (ридж/Procrustes на train) **не делался**: идея
зафиксирована в README §6 как нереализованная, решение от 2026-10-28 — отложить.

## 20. Четвёртый размер: Qwen 3.5 9B в Q4_K_M (шаг 13)

`Qwen3.5-9B-Q4_K_M.gguf` (5.68 ГБ, arch `qwen35`, **32 слоя, D = 4096**, `l_out` на всех
32 слоях). Вход, флаги и протокол пробы — те же, что у 0.8B / 2B / 4B (§18, §19).
Задача та же: довести серию размеров до предела, который влезает в 6 ГБ VRAM.

### 20.1. Прогон

Пилот 200 (`out/pilot_qwen35_9b_q4`): 200/200 ok, 0 `truncated`, `retokenize mismatch`
0/200, ответы осмысленные. Полный прогон (`out/main_qwen35_9b_q4`):

| | значение |
|---|---|
| итог | **6000/6000 ok, 0 skipped, `truncated` 0** |
| время | **2031.1 s (33.9 мин), 2.95 ex/s** |
| `retokenize mismatch` | 11/6000 (0.2 %) |
| массивы | `[6000, 32, 4096]` f16, по 1500 МБ |
| `verify_run.py --full` | **0 failed, 0 warnings** (дубль `bb_956` — как во входе) |

**Подбор `-ngl`.** При 5.68 ГБ весов в 6 ГБ VRAM авто-подбор отказывается
(`failed to fit params to free device memory`), но `-ngl 99` работает: пик VRAM 5860 МиБ
из 6144, вытеснения нет. Замер на 50 строках:

| `-ngl` | пик VRAM | скорость |
|---|---|---|
| **99** | 5860 МиБ | **2.90 ex/s** |
| 30 | 5878 МиБ | 1.96 ex/s |
| 26 | 5383 МиБ | 1.31 ex/s |
| 22 | 4853 МиБ | 1.01 ex/s |

То есть 9B влезает в 6 ГБ целиком, и на CPU-слоях только медленнее. Снижать `-ngl`
не нужно.

**Аномалия измерения, важная для будущих прогонов.** Первый запуск пилота шёл через
`| Tee-Object` и дал 232.2 s (0.86 ex/s) с деградацией по ходу: первые 10 строк 2.5 ex/s,
к 100-й — 0.91. Это не VRAM и не троттлинг: те же 200 строк с перенаправлением в файл
(`> log 2>&1`) прошли за **74.5 s (2.68 ex/s)**. Вывод: полные прогоны выводить только
в файл, `Tee-Object` замедляет процесс в 3 раза (блокировка на записи консоли). На
массивы это не влияет (детерминизм сохранён), но оценка времени по такому пилоту
завышается втрое.

### 20.2. Разметка

| | 9B Q4 |
|---|---|
| strict | 0.572 (3430/6000) |
| relaxed | 0.575 (3451) |
| `filtered_a` | 0.567 (3402) |
| `filtered_b` | **0.570 (3420)** |
| null | **0** |
| strict по сплитам | 0.583 / 0.554 / 0.555 |
| `filtered_b` по сплитам | 0.582 / 0.552 / 0.553 |
| `strict ≠ relaxed` | 21 строка |

Пустых, обрезанных и вырожденных ответов нет вообще — впервые в серии. Ответы самые
короткие: в среднем 11.7 символа (медиана 10, p95 23) против 29.5 у 0.8B, 20.7 у 2B и
13.5 у 4B; всё это потому, что 9B не уходит в развёрнутый текст и не упирается в 64
токена (0 обрезок против 20–22 у предыдущих размеров). Доля правильных растёт до 0.570 —
это уже уровень «больше половины», чего в серии не было.

Пилот (`out/pilot_qwen35_9b_q4`, разметка добавлена) даёт ту же картину: strict 0.605 на
200 строках против 0.572 на полном наборе, `null` 0, одна строка `strict ≠ relaxed`.

### 20.3. Baseline

| | val | test |
|---|---|---|
| logprob | 0.859 [0.838, 0.878] | 0.870 [0.850, 0.890] |
| entropy | 0.869 [0.849, 0.888] | 0.877 [0.856, 0.897] |
| test AUPRC (logprob / entropy) | — | 0.881 / 0.888 |
| ECE после калибровки (logprob / entropy) | 0.055 / 0.052 | 0.049 / 0.045 |
| сырая `exp(lp0)`, ECE | 0.040 | 0.070 |
| базовая доля | 0.552 | 0.553 |

**Baseline вышел на плато:** у 4B было 0.870 / 0.878, у 9B — 0.870 / 0.877, то есть
прирост от 4B к 9B нулевой. Это первое наблюдение §20.6: первый токен у моделей 4B и 9B
одинаково хорошо связан с правильностью ответа.

### 20.4. Проба

Протокол не менялся: сетка λ `{100, 300, 1000, 3000}`, точки A/C, все 32 слоя, пара
(точка, слой) по max val AUROC, тест считается один раз.

| | 9B Q4 |
|---|---|
| средний val AUROC по λ (100/300/1000/3000) | 0.794 / 0.809 / 0.822 / **0.826** |
| λ | **3000** |
| пара | **C, слой 16** (ровно 50 % глубины) |
| val AUROC / AUPRC | 0.9120 [0.896, 0.926] / 0.9278 |
| **test AUROC** | **0.9172 [0.901, 0.933]** |
| test AUPRC | 0.9271 |
| test Brier / ECE | 0.1143 / 0.0240 (после Platt 0.0149) |
| контроль: перемешанные метки | 0.531 ± 0.027 (5 перестановок) |
| контроль: слой 0, A / C | 0.655 / 0.661 |

Кривая по слоям обычная: у точки C максимум в середине (16: 0.9120, соседи 17: 0.9112,
15: 0.9087, 18: 0.9076 — разброс 0.004), у точки A — в глубоких слоях (30: 0.8637,
31: 0.8622), на слое 0 минимум (C 0.6162, A 0.6531). То есть A устойчиво хуже C на
0.05–0.06, как и у 2B/4B.

Чувствительность к λ низкая: в паре C/16 λ = 1000 даёт 0.9098 против 0.9120 при 3000
(разница 0.002) — сетка держится, менять её не нужно. Разрыв val→test положительный
(+0.005), как у Gemma (§14): тест чуть выше val, интервалы перекрываются; аномалией не
считаем и тест не пересчитываем.

Детерминизм: повторный запуск (`probe2.json`, `probe2_curve.csv`, `probe2_scores.csv`)
даёт побайтово те же `probe_curve.csv` и `probe_scores.csv` (SHA-256 совпадают).

### 20.5. Сравнение с другими размерами

Парный bootstrap по вопросам (`compare_probes.py`, 1000 итераций, seed 42):

| сравнение | ΔAUROC test [95 % CI] | доля bootstrap, где B лучше | ΔAUROC val [95 % CI] |
|---|---|---|---|
| 9B vs 0.8B | **+0.0472 [+0.016, +0.081]** | **1.00** | +0.0283 [−0.003, +0.064] |
| 9B vs 2B | **+0.0307 [+0.007, +0.055]** | **0.99** | +0.0172 [−0.006, +0.041] |
| **9B vs 4B** | **−0.0002 [−0.016, +0.017]** | 0.53 | −0.0100 [−0.027, +0.007] |

**9B достоверно лучше 0.8B и 2B, но не отличается от 4B** — на тесте разница ровно
−0.0002, то есть ноль, и в 53 % bootstrap-выборок 9B выше. На val 4B даже чуть выше
(0.9218 против 0.9120), но интервал задевает ноль.

Метки при этом разные, и это важно для интерпретации: на test 9B права там, где 4B
ошибается, на 186 строках, а 4B права там, где 9B ошибается, — на 54 (согласие 80 %).
То есть 9B объективно отвечает лучше (доля правильных 0.570 против 0.454), но проба
этот прирост **не отражает**: обе упёрлись в 0.917–0.918 AUROC. ΔAUPRC между ними
(+0.025) отражает рост базовой доли, а не качество ранжирования.

### 20.6. Насыщение: что происходит между 4B и 9B

Главный результат шага 13 — не «9B лучше», а **плато по обеим величинам сразу**:

| | 0.8B Q4 | 2B Q4 | 4B Q4 | 9B Q4 |
|---|---|---|---|---|
| слоёв × D | 24 × 1024 | 24 × 2048 | 32 × 2560 | 32 × 4096 |
| время | 731.8 s (8.20 ex/s) | 866.0 s (6.93 ex/s) | 1483.5 s (4.04 ex/s) | 2031.1 s (2.95 ex/s) |
| `truncated` | 22 | 32 | 20 | **0** |
| `filtered_b` | 0.118 | 0.253 | 0.454 | **0.570** |
| baseline test: logprob / entropy | 0.680 / 0.717 | 0.783 / 0.794 | 0.870 / 0.878 | **0.870 / 0.877** |
| проба: λ / пара | 1000 / C 13 | 3000 / C 12 | 3000 / C 15 | **3000 / C 16** |
| глубина слоя | 57 % | 52 % | 48 % | **50 %** |
| val AUROC | 0.8838 | 0.8948 | **0.9218** | 0.9120 |
| **test AUROC** | 0.8709 | 0.8855 | **0.9180** | **0.9172** |
| прирост пробы над энтропией первого токена | **+0.154** | **+0.092** | **+0.040** | **+0.040** |

Как это читать:

- **Доля правильных продолжает расти** (0.118 → 0.253 → 0.454 → 0.570), то есть модель
  становится лучше по существу. Растёт и baseline, но только до 4B.
- **Качество пробы упирается в ~0.918 на 4B и дальше не растёт** (0.9180 → 0.9172).
- **Прирост пробы над baseline перестал падать:** +0.040 и у 4B, и у 9B. Раньше он
  сокращался вдвое на каждом шаге (+0.154 → +0.092 → +0.040), потому что baseline
  догонял. У 9B baseline не вырос, и прирост остался тем же. Это означает, что
  4B/9B находятся в одном режиме: первый токен объясняет примерно столько же, сколько
  объяснял, а скрытые состояния добавляют фиксированные ~0.04.
- **Ошибки 4B и 9B разные, а проба их ранжирует одинаково.** 9B отвечает правильно на
  186 test-строках, где 4B ошибается, 4B — на 54, где ошибается 9B; при этом AUROC проб
  совпадает до третьего знака. Значит потолок ~0.92 — свойство **признака** (один
  вектор скрытого состояния в точке C, линейная логистическая регрессия), а не объёма
  знаний модели: дальнейший рост числа правильных ответов не превращается в лучший
  ранжирующий сигнал. Проверить это дальше можно только другим признаком (несколько
  слоёв/позиций, нелинейность) — это выходит за замороженный протокол и не делалось.

### 20.7. Кросс-скоры с 9B (тест)

| источник скора → метка | родная проба | кросс AUROC [95 % CI] | контроль «метка → метка» | Δ над контролем | согласие меток |
|---|---|---|---|---|---|
| 9B → 0.8B | 0.9181 | 0.6667 [0.639, 0.696] | 0.5945 | +0.072 | 0.553 |
| 0.8B → 9B | 0.8709 | 0.7708 [0.737, 0.806] | 0.7129 | +0.058 | 0.553 |
| 9B → 2B | 0.9162 | 0.7379 [0.712, 0.765] | 0.6908 | +0.047 | 0.661 |
| 2B → 9B | 0.8855 | 0.8145 [0.790, 0.840] | 0.7470 | +0.068 | 0.661 |
| 9B → 4B | 0.9178 | 0.8148 [0.792, 0.838] | 0.8093 | **+0.006** | 0.800 |
| 4B → 9B | 0.9180 | 0.8605 [0.839, 0.883] | 0.8097 | +0.051 | 0.800 |

Выводы §19.8 сохраняются: кросс AUROC нельзя сравнивать с 0.5, корректный нулевой
уровень — контроль «метка → метка» (0.59–0.81), и сверх него скор даёт +0.006…+0.072.

Две новые детали:

- **Пара 4B ↔ 9B почти вырождена:** согласие меток 0.800, корреляция скоров 0.739,
  а скор 9B против метки 4B даёт 0.8148 при контроле 0.8093 — то есть +0.006, ничего.
  Скор 9B практически избыточен относительно знания «права ли 4B».
- **Асимметрия здесь развернулась:** 9B → 0.8B (0.667) хуже, чем 0.8B → 9B (0.771).
  Это не про силу моделей, а про базовую долю цели: у 9B положительных 0.55, у 0.8B —
  0.13, и ранжировать редкие положительные чужой меткой труднее. Поэтому направление
  кросс-сравнений между разными парами размеров читать нельзя; годятся только пары
  «скор → своя метка / чужая метка» внутри одного направления.

### 20.8. Артефакты

```
llama-hidden-states.exe -m Qwen3.5-9B-Q4_K_M.gguf \
  --batch data\main.jsonl --generate --max-new-tokens 64 --max-len 1024 \
  --layers all -ngl 99 --out out\main_qwen35_9b_q4      # 2031.1 s

python label.py --meta out\main_qwen35_9b_q4\meta.jsonl --out out\main_qwen35_9b_q4\labels.csv
python verify_run.py --dir out\main_qwen35_9b_q4 --full --json out\main_qwen35_9b_q4\verify.json
python baseline.py --meta out\main_qwen35_9b_q4\meta.jsonl \
  --labels out\main_qwen35_9b_q4\labels.csv --out out\main_qwen35_9b_q4\baseline.json \
  --scores-out out\main_qwen35_9b_q4\baseline_scores.csv
python probe.py --hidden-a out\main_qwen35_9b_q4\hidden_a.npy \
  --hidden-c out\main_qwen35_9b_q4\hidden_c.npy --meta out\main_qwen35_9b_q4\meta.jsonl \
  --labels out\main_qwen35_9b_q4\labels.csv --out out\main_qwen35_9b_q4\probe.json \
  --curve-out out\main_qwen35_9b_q4\probe_curve.csv \
  --scores-out out\main_qwen35_9b_q4\probe_scores.csv          # ~37 мин CPU
python compare_probes.py --a out\main_qwen35_4b_q4\probe_scores.csv \
  --b out\main_qwen35_9b_q4\probe_scores.csv --split test \
  --out out\compare_qwen35_4b_qwen35_9b_test.json
```

Каталог `out/main_qwen35_9b_q4/`: `probe.json`, `probe_curve.csv`, `probe_scores.csv`,
`probe_model.npz`, `baseline.json`, `baseline_scores.csv`, `labels.csv`, `meta.jsonl`,
`config.json`, `verify.json`, плюс `probe2*` (проверка детерминизма). Сравнения:
`out/compare_qwen08q4_qwen35_9b_{test,val}.json`,
`out/compare_qwen35_2b_qwen35_9b_{test,val}.json`,
`out/compare_qwen35_4b_qwen35_9b_{test,val}.json` (6 файлов). Кросс-скоры: 12 файлов
`out/cross_*_qwen35_9b_*` / `out/cross_qwen35_9b_to_*`. Пилот: `out/pilot_qwen35_9b_q4`.
