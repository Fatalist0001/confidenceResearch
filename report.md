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
