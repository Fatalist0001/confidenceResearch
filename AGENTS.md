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

Полные числа и таблицы — в `report.md`, раздел 7.

## Чего не делать

- Не менять v1-ветку и её формат выхода (`[L, D]` + `hidden_states.json`).
- Не править ядро llama.cpp — инструмент написан только на публичном API
  (`llama.h`, `common/*.h`, включая `chat.h`, `build-info.h`).
- Не добавлять многопоточность и лишние флаги (договорённость: «не усложнять»).
- Не менять фиксированные определения точек A и C.
