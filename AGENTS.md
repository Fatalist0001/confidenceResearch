# AGENTS.md

Рабочий контракт проекта `llama-hidden-states`. Полная история решений, инцидентов и
таблиц — в `docs/history/AGENTS-full-2026-11-02.md`; здесь только то, что нужно, чтобы
ничего не сломать.

**Перед первой командой прочитайте `docs/START-HERE.md`.** Там порядок чтения, какие
прогоны цитировать, список заблокированных действий и рецепты.

---

## 1. Что это

CLI-инструмент извлекает скрытые состояния всех доступных слоёв GGUF-модели в двух
точках ответа (A и C) поверх llama.cpp. По одному вектору слоя обучается L2-логистическая
проба; её AUROC сравнивается с baseline (logprob и энтропия первого токена).

Два режима:

- **v1, одиночный** — поведение и формат заморожены, не менять;
- **v2, пакетный** — массовые прогоны, `--batch input.jsonl`, resume, `--generate`.

Состояние проекта в машинночитаемом виде:

| файл | что |
|---|---|
| `RUNS.json` | реестр прогонов и числа из артефактов |
| `profiles.json` | флаги и ожидания по семействам моделей |
| `PROJECT_STATE.json` | бинарники, протокол, блокировки, guard-правила |

Эти три файла — источник истины. README и report могут отставать; реестр генерируется
из артефактов командой `scripts/build_runs.py` и проверяется `scripts/check_state.py`.
Числа в `RUNS.json` руками не переписывать: для нового прогона добавить только
provenance-строку в каталог генератора и запустить его.

---

## 2. Где живёт код

| путь | что |
|---|---|
| `llama.cpp-hiddenstates-new\` | master `785322d17`, build 12033 — **все новые прогоны** |
| `llama.cpp-hiddenstates\` | старый пин `d4cff114c`, build 9869 — только воспроизведение старых `.npy` |
| `llama.cpp-hiddenstates*\examples\hidden-states\hidden-states.cpp` | вся программа, одинаковая в обеих копиях |
| `llama.cpp-hiddenstates*\examples\hidden-states\CMakeLists.txt` | таргеты `llama-hidden-states`, `hs-prompt-dump` |
| `llama.cpp-hiddenstates*\examples\hidden-states\prompt_dump.cpp` | диагностика промпта без загрузки весов |
| `llama.cpp-hiddenstates*\examples\hidden-states\load_example.py` | просмотр результатов |
| `llama.cpp-hiddenstates*\examples\CMakeLists.txt` | изменён на одну строку: `add_subdirectory(hidden-states)` |
| `pipeline\` | Python: probe, baseline, label, verify, gate, preflight, сборщики промптов |
| `scripts\` | `run.ps1`, `bootstrap.ps1`, `doctor.py`, `check_state.py` |
| `kaggle\` | ноутбуки и запуск на Kaggle |

В git внутри копий llama.cpp отслеживаются **только** пять файлов:

- `examples/CMakeLists.txt`
- `examples/hidden-states/CMakeLists.txt`
- `examples/hidden-states/hidden-states.cpp`
- `examples/hidden-states/load_example.py`
- `examples/hidden-states/prompt_dump.cpp`

Всё остальное (`src\`, `common\`, `vendor\`, `build\`) игнорируется. Поэтому **чистый
клон не соберётся**, пока не выполнен:

```powershell
powershell -File scripts\bootstrap.ps1
```

Скрипт скачивает tar-архивы нужных коммитов с codeload (GitHub не отдаёт fetch
произвольного SHA) и не перезаписывает наши пять файлов. Вложенный `.git` в копиях
удалён: иначе родительский репозиторий считал бы их субмодулями.

---

## 3. Бинарники

| копия | commit | build | `--version` |
|---|---|---|---|
| active `llama.cpp-hiddenstates-new` | `785322d17` | 12033 | `build 12033, commit 785322d17` |
| legacy `llama.cpp-hiddenstates` | `d4cff114c` | 9869 | `version: 9869 (d4cff114c)` |

Сборка: `powershell -File scripts\bootstrap.ps1 -Build`; ручные CMake-команды — в
`docs/SETUP.md`. Commit/build задавать явно: у архивных копий нет `.git`. После сборки
`python scripts\doctor.py` должен дать `0 failed, 0 warnings`.

---

## 4. Инварианты (нарушать нельзя)

1. **Точка A** — последний токен промпта, отрендеренного чат-шаблоном с
   `add_generation_prompt=true`; состояние, из которого выдаётся первый токен ответа.
   **Точка C** — последний содержательный токен ответа, без EOG/конца реплики.
2. Промпт и ответ токенизируются **раздельно** (`parse_special=true`), затем id
   склеиваются.
3. Промпт токенизируется с `add_special = llama_vocab_get_add_bos(vocab)`. llama.cpp
   сам вырезает `<bos>`, который печатает шаблон, и ждёт его от токенизатора — так же
   делает `llama-cli`.
4. Состояния считаются **отдельным проходом** по «промпт + ответ» (A и C из одного
   прохода). Генерация идёт другим проходом. Перед проходом состояний обязателен
   `llama_memory_clear(mem, true)`.
5. `n_batch = n_ubatch = --max-len`, и `n_tokens <= max_len`. Это гарантирует один
   ubatch и соответствие индекса строки индексу токена. Слишком длинные примеры
   **пропускаются** с причиной в meta, прогон не падает.
6. `logits=true` минимум на одной позиции, иначе `GGML_ASSERT(n_outputs >= 1)`.
7. `params.warmup = false`.
8. Колбэк `cb_eval`: в фазе `ask` возвращать `true` **только** для `l_out-*`; в фазе
   `ask=false` всегда `true`, иначе `break` оборвёт граф.
9. Свои флаги вынимаются из `argv` **до** `common_params_parse`.
10. Жадный выбор токена (`hs_argmax`) пропускает `suppress_tokens` модели: с
    2026-07-29 llama.cpp перенёс их из графа в сэмплер, а сэмплера у нас нет.

Менять определения A/C, порядок проходов, токенизацию и батчевые параметры нельзя.
Это ломает сравнимость со всеми существующими прогонами.

---

## 5. Пакетный режим

Флаги задаёт профиль; базовый вызов: `llama-hidden-states -m <gguf> --batch <jsonl> --out <dir> --generate`.
Вход содержит `{"id","question","answer"?}`; неизвестные
ключи переходят в `meta.jsonl`. Выход:
`hidden_{a,c}.npy` float16 `[N,L_sel,D]`, `meta.jsonl`, `config.json`.

Правила:

- **не передавать `-b/-ub`**: пакетный режим сам ставит их равными `--max-len`;
- **не писать логи через `Tee-Object`**: на 9B это замедляло прогон втрое; логировать
  в файл (`scripts\run.ps1` делает это правильно);
- `--out` **никогда** не должен указывать в существующий каталог прогона: будет resume
  и перезапись строк. Новый прогон — новый тег.

Формат v1 (`[L, D]` + `hidden_states.json`) заморожен и не меняется.

---

## 6. Замороженный протокол пробы

Полное описание — `docs/START-HERE.md#frozen-probe-protocol`, `README.md` §6,
`pipeline/probe.py`.

- метка `correct_filtered_b`;
- признак — один вектор одного слоя, точка A или C;
- стандартизация по train;
- L2-логистика (Newton/IRLS), λ ∈ `{100, 300, 1000, 3000}`;
- λ выбирается по **среднему val AUROC**, точка/слой — по **val AUROC** при этой λ;
- test считается **один раз**; по test ничего не перевыбирается;
- bootstrap 1000, seed 42;
- веса сохраняются в `probe_model.npz`.

Проверка детерминизма — повторный запуск в `probe2.*`; SHA-256 `probe_curve.csv` и
`probe_scores.csv` должны совпасть. Оригинальные `probe.*` при этом не перезаписываются.

Ориентиры (не цели): shuffled AUROC ≈ 0.5, test обычно 0.86–0.94, победитель чаще C
около середины глубины. Подгонять λ/слой/split под эти числа запрещено.

---

## 7. Рабочий цикл: только через раннер

Не копируйте длинные команды из старых записей. Используйте `scripts/run.ps1`: он
читает флаги из `profiles.json`, пишет логи в файл и отказывает в небезопасных
переходах.

Порядок стадий: doctor → inputs → prompt → pilot → main → label → verify → baseline →
probe → probe2 → compare; точные команды — в `docs/START-HERE.md`. Для полного
нового прогона допустим `-Stage all`; сначала полезно проверить его с `-DryRun`.

Раннер гарантирует:

- не перезаписывает каталог, лог, `labels.csv`, `verify.json`, `baseline.json`,
  `probe*.json`;
- `pilot/main` требуют успешный prompt-record того же тега; сверяются hash входа,
  профиль, модель и binary commit/build;
- фактический `--version` бинарника перед prompt/generation должен совпасть с
  `PROJECT_STATE.json`;
- pilot автоматически размечается и проходит повторный class-balance gate; отдельная
  стадия `label` предназначена для main;
- `main` не стартует без `pilot_<tag>/gate.json` с `passed=true` и без совпадения модели;
- `probe` не стартует без `verify.json` с `fails=0`;
- `prompt` сверяет `add_bos` и первый токен с `profiles.json` и проверяет, что сборщик
  промпта идемпотентен;
- заблокированные действия требуют `-ConfirmDeferred`;
- `-b/-ub` не передаются.

Сырые команды допустимы только для отладки, не для «настоящих» прогонов.
При `-InputFile` обязателен `-QuestionsFile` с исходными неотрендеренными вопросами,
иначе `filtered_a/b` могли бы считаться по harmony/final-cue промпту.

---

## 8. Особые режимы

- **BOS:** при `add_bos=1` первый токен обязан равняться BOS id; иначе остановиться.
  `main_gemma4e2b`, старый E4B и Nemotron v2 — валидные исторические `legacy_no_bos`.
- **gpt-oss-20b:** профиль сам строит harmony `final`; обязательны `--no-template`,
  `-ngl 11`, `--batch-rows 8`, инструкция ровно одна.
- **LFM2.5-2.6B:** профиль строит cue `Final answer:` и использует `--no-template`;
  A — последний токен cue.
- **LFM2.5-1.2B-Thinking:** обычный прогон запрещён (200/200 truncated); сначала
  зафиксировать reasoning/final budgets, извлечение final и C.
- **Nemotron:** нормально 17 `l_out` из 42 и `verify warns=1`; нули/skipped — дефект.
- **Qwen3.5-9B:** только `-ngl 99`, лог в файл (пик ~5860 MiB).
- **Gemma 12B+:** только Kaggle; quota → pilot → бюджет → явное разрешение на main.

---

## 9. Заблокированные действия

Список — `PROJECT_STATE.json → authorization.blocked_actions`; раннер его соблюдает.

| id | что нельзя |
|---|---|
| `run:lfm25-1.2b-thinking` | обычный прогон Thinking |
| `run:lfm3-sampler-diagnostic` | новый полный прогон LFM3 до диагностики сэмплера |
| `kaggle:full-run` | полный прогон на Kaggle без пилота и бюджета |
| `protocol:change` | правка λ/splits/точек/метки |
| `artifacts:rewrite-or-delete` | перезапись или удаление артефактов |

`-ConfirmDeferred` существует, но применять его можно только по явной просьбе
пользователя.

`HALLUCINATION_PLAN.md` и `QUANT_TRANSFER_PLAN.md` — планы, а не задачи. В их начале
стоит баннер `NOT AUTHORIZED`.

---

## 10. Сохранность данных

Прогон 6000 строк стоит от 20 минут до 5 часов и не воспроизводится «на лету».

- каждый прогон — в свой `pipeline\out\{main,pilot}_<tag>`, тег не повторять;
- `meta.jsonl`, `labels.csv`, `baseline.*`, `probe.*`, `hidden_*.npy`, `verify.json`
  считать окончательными;
- новые вычисления — в новые имена (`probe2.*`, новый тег);
- `.npy` не в git и живут только на диске; `pipeline\out\` не чистить;
- невалидные каталоги (`main_gemma4_12b`, `main_gptoss20b_dupinstr`,
  `main_nemotron3nano4b`) и остановленный `main_gptoss20b` не удалять: это улики.

После любого прогона сверять:

- строк в `meta.jsonl` == ожидаемому N;
- `ok + skipped` совпадает;
- формы массивов == `[N, n_layers_selected, n_embd]` из `config.json`;
- `probe_scores.csv` содержит ровно val + test.

---

## 11. Проверки после правок

| проверка | команда | ожидание |
|---|---|---|
| окружение | `python scripts\doctor.py` | 0 failed |
| воспроизводимость реестра | `python scripts\build_runs.py --check` | current |
| реестр и документация | `python scripts\check_state.py` | 0 failed |
| unit-тесты | `python -m unittest discover -s pipeline\tests -v` | OK |
| синтаксис PowerShell | парсер `[Parser]::ParseFile` | ошибок нет |
| JSON | `ConvertFrom-Json` на трёх файлах | валидно |
| лишние файлы | `git status --short` | только ожидаемые |

Перед коммитом: `git diff --check`, глазами просмотреть diff, убедиться, что не
изменились существующие артефакты.

---

## 12. Коммиты

- Коммиты — только от имени `Fatalist0001`.
- **Не добавлять** `Assisted-by:`, `Co-authored-by:`, `Generated-by:` и любые
  упоминания AI-ассистентов.
- Коммит и push — только по явной просьбе пользователя.

---

## 13. Чего не делать

- Не менять v1-ветку и её формат (`[L, D]` + `hidden_states.json`).
- Не править ядро llama.cpp: инструмент использует только публичный API
  (`llama.h`, `common/*.h`, `chat.h`, `build-info.h`).
- Не добавлять многопоточность и лишние флаги.
- Не удалять и не перезаписывать `out\*`, `.npy`, `meta.jsonl`, `labels.csv`,
  `probe.*`, `baseline.*` без явной просьбы.
- Не запускать `--out` в существующий каталог.
- Не менять определения A и C.
- Не менять λ-сетку, splits, точки и метку ради результата.
- Не запускать полный прогон без пилота и гейта.
- Не жечь GPU-часы Kaggle вслепую.
- Не считать план задачей.

---

## 14. Где подробности

| тема | файл |
|---|---|
| порядок работы, рецепты, playbook | `docs/START-HERE.md` |
| устройство окружения и сборки | `docs/SETUP.md` |
| научные результаты | `README.md` |
| полный отчёт и таблицы проверок | `report.md` |
| реестр прогонов | `RUNS.json` |
| флаги по моделям | `profiles.json` |
| бинарники, протокол, блокировки | `PROJECT_STATE.json` |
| история: инциденты, старые команды, все детали | `docs/history/AGENTS-full-2026-11-02.md` |
| Kaggle | `kaggle/README.md` |
