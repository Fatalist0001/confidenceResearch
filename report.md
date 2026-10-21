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
