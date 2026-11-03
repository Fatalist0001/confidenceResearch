# SETUP: как поднять окружение с нуля

Этот репозиторий **не самодостаточен по исходникам llama.cpp**: в git лежат только наши
пять файлов в каждой из двух копий и одна строка в `examples/CMakeLists.txt`. Чистый
клон не соберётся, пока не выполнен bootstrap.

---

## 1. Что нужно установить

| компонент | версия / примечание |
|---|---|
| Windows | проект проверен на Windows 10/11, PowerShell 5.1 |
| Git | любой свежий, с доступом к github.com |
| Python | **3.11+** (проверено на 3.11.9) |
| CMake | 3.20+ |
| Visual Studio 2022 | «Desktop development with C++», MSVC 19.39 |
| CUDA Toolkit | 12.x, если нужен GPU-бэкенд (`GGML_CUDA=ON`) |
| NVIDIA driver | достаточный для вашей CUDA |
| `tar` | встроен в Windows 10+ (`C:\Windows\System32\tar.exe`) |

Опционально: `uv` — если хотите ставить Python-зависимости из `pipeline/uv.lock`.

Проверить:

```powershell
git --version
python --version
cmake --version
tar --version
nvidia-smi
```

---

## 2. Bootstrap чистого клона

```powershell
git clone https://github.com/Fatalist0001/confidenceResearch
cd confidenceResearch

# посмотреть план, ничего не записывая
powershell -File scripts\bootstrap.ps1 -DryRun

# скачать исходники обоих пинов и создать pipeline\.venv
powershell -File scripts\bootstrap.ps1

# то же + собрать CUDA-бинарники
powershell -File scripts\bootstrap.ps1 -Build

# только CPU-сборка (если нет CUDA)
powershell -File scripts\bootstrap.ps1 -Build -NoCuda
```

Что делает скрипт:

1. Проверяет, есть ли в `llama.cpp-hiddenstates*` файлы `src/llama-context.cpp` и
   `common/chat.cpp`. Если нет — скачивает tar-архив нужного коммита с
   `https://codeload.github.com/ggml-org/llama.cpp/tar.gz/<sha>` (пин ≈ 33.5 МБ, master
   ≈ 37.1 МБ) и распаковывает его **поверх**, исключая наши пять файлов.
2. Проверяет, что наши пять файлов на месте и что
   `add_subdirectory(hidden-states)` встречается ровно один раз.
3. Создаёт `pipeline\.venv` и ставит `pipeline\requirements.txt`.
4. С `-Build` конфигурирует CMake и собирает `llama-hidden-states` и `hs-prompt-dump`.

Флаги:

| флаг | смысл |
|---|---|
| `-Only active` / `-Only legacy` | только одна копия llama.cpp |
| `-SkipFetch` | не скачивать, только проверить |
| `-NoVenv` | не трогать Python-окружение |
| `-NoCuda` | `-DGGML_CUDA=OFF` |
| `-DryRun` | печатать команды, ничего не писать |

Скрипт **никогда** не удаляет `build\` и `pipeline\out\`. Если дерево уже есть, он
оставляет его как есть.

> Если `git fetch origin <sha>` когда-нибудь снова покажется хорошей идеей: GitHub
> отказывает в fetch произвольного SHA (`fatal: couldn't find remote ref d4cff114c`).
> Поэтому используется codeload-архив.

---

## 3. Ручная сборка (если bootstrap не подходит)

Активная копия (master):

```powershell
$LL = "D:\VScode_pr\confidenceResearch\llama.cpp-hiddenstates-new"
cmake -S $LL -B $LL\build -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=ON `
  -DLLAMA_BUILD_COMMIT=785322d17 -DLLAMA_BUILD_NUMBER=12033
cmake --build $LL\build --config Release `
  --target llama-hidden-states --target hs-prompt-dump -- /m
```

Пин:

```powershell
$LL = "D:\VScode_pr\confidenceResearch\llama.cpp-hiddenstates"
cmake -S $LL -B $LL\build -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=ON `
  -DLLAMA_BUILD_COMMIT=d4cff114c -DLLAMA_BUILD_NUMBER=9869
cmake --build $LL\build --config Release `
  --target llama-hidden-states --target hs-prompt-dump -- /m
```

Проверка версии:

```powershell
& .\llama.cpp-hiddenstates-new\build\bin\Release\llama-hidden-states.exe --version
# build 12033, commit 785322d17

& .\llama.cpp-hiddenstates\build\bin\Release\llama-hidden-states.exe --version
# version: 9869 (d4cff114c)
```

`--version` у старого пина печатает другой формат — это нормально, `doctor.py`
понимает оба.

---

## 4. Python-окружение

```powershell
python -m venv pipeline\.venv
& .\pipeline\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\pipeline\.venv\Scripts\python.exe -m pip install -r pipeline\requirements.txt
```

Зависимости: `numpy`, `datasets`. Больше ничего не нужно для `probe.py`, `baseline.py`,
`verify_run.py`, `label.py`.

Через `uv` (опционально):

```powershell
cd pipeline
uv sync
```

`pyproject.toml` и `uv.lock` держат те же две зависимости.

---

## 5. Проверка окружения

```powershell
& .\pipeline\.venv\Scripts\python.exe scripts\doctor.py
# или через раннер:
powershell -File scripts\run.ps1 -Stage doctor -Profile qwen
```

`doctor.py` проверяет:

- Python ≥ 3.11 и пакеты `numpy`, `datasets`;
- наличие venv;
- наличие обоих бинарников и совпадение `--version` с `PROJECT_STATE.json`;
- наличие `hs-prompt-dump`;
- исходники обеих копий llama.cpp;
- побайтовую идентичность наших пяти файлов в обеих копиях;
- число строк `data/main.jsonl` (6000) и `data/pilot.jsonl` (200);
- запись в `pipeline/out`;
- свободное место и видимость CUDA GPU.

Ожидаемый результат: `doctor: 0 failed, 0 warnings`.

Тесты и проверка реестра:

```powershell
& .\pipeline\.venv\Scripts\python.exe -m unittest discover -s pipeline\tests -v
& .\pipeline\.venv\Scripts\python.exe scripts\build_runs.py --check
& .\pipeline\.venv\Scripts\python.exe scripts\check_state.py
```

`check_state.py` сверяет `RUNS.json` с диском: строки meta, формы, метки, baseline,
probe, каноничность, заблокированные id, идентичность файлов в двух копиях. Он
read-only; при расхождении правится **файл реестра**, а не артефакт.

---

## 6. Данные

`pipeline/data/main.jsonl` (6000 строк) и `pipeline/data/pilot.jsonl` (200 строк) уже в
git. Пересобирать их без причины не нужно.

Если нужно собрать заново из TriviaQA:

```powershell
cd pipeline
& .\.venv\Scripts\python.exe prepare_data.py
```

После пересборки обязательно:

```powershell
& .\.venv\Scripts\python.exe preflight.py --input data\main.jsonl `
  --expect-rows 6000 --require-instruction
```

Ожидание: 6000 строк, ровно одна инструкция в каждом вопросе, дубль id только `bb_956`.

---

## 7. Kaggle (GPU вне этой машины)

Локальный 6 ГБ GPU не тянет 12B+; для таких моделей используется Kaggle. Нужен токен
`KAGGLE_API_TOKEN` в `.env` (файл в `.gitignore`), CLI `kaggle` и пройденная верификация
телефона — без неё GPU молча не выдаётся.

```powershell
cd D:\VScode_pr\confidenceResearch
$env:KAGGLE_API_TOKEN = ((Get-Content .env |
  Select-String '^KAGGLE_API_TOKEN=').Line -replace '^KAGGLE_API_TOKEN=','').Trim()

& 'C:\Windows\Temp\opencode\kgvenv\Scripts\kaggle.exe' quota
powershell -File kaggle\gemma4_12b\push.ps1
powershell -File kaggle\watch.ps1 -Kernel odinodinodin/hs-gemma4-12b -IntervalSec 60
```

Правила: сначала пилот 20–50 строк, потом оценка бюджета, только потом 6000 строк.
Сборку и `probe.py` делать в CPU-сессии — они не тратят GPU-квоту. Подробности —
`kaggle/README.md` и `docs/history/AGENTS-full-2026-11-02.md`.

---

## 8. Грабли Windows

- **CRLF.** В репозитории включён `core.autocrlf=true`, поэтому файлы в рабочем дереве
  могут отличаться от файлов в свежем клоне **только переводами строк**. Содержимое
  одинаковое; при сравнении хешей учитывайте это.
- **Кодировка консоли.** Старые логи и `--version` могут выглядеть кракозябрами в
  PowerShell. Это отображение, а не порча файла; читайте файлы в UTF-8.
- **Долгие прогоны и `Tee-Object`.** Не пропускайте вывод 6000-строчного прогона через
  `Tee-Object`: на 9B это замедляло пилот втрое. Раннер сам пишет stdout/stderr в один
  файл через `Start-Process`.
- **`2>&1` при `$ErrorActionPreference='Stop'`.** PowerShell 5.1 превращает stderr
  нативного процесса в `NativeCommandError`. Поэтому раннер не использует пайпы для
  логов.
- **Антивирус/индексатор.** Запись в `pipeline/out` иногда секунду блокируется;
  `doctor.py` повторяет пробную запись несколько раз, прежде чем сообщить об ошибке.

---

## 9. Что делать, если что-то не собирается

| ошибка | что проверить |
|---|---|
| `missing our files in llama.cpp-*` | вы в чистом клоне без bootstrap; запустите `scripts\bootstrap.ps1` |
| `add_subdirectory(hidden-states)` не 1 раз | восстановите `examples/CMakeLists.txt` из git, не правьте руками |
| CMake не находит CUDA | поставьте CUDA Toolkit или собирайте с `-NoCuda` |
| `binary not found` в раннере | `scripts\bootstrap.ps1 -Build` или ручная сборка выше |
| `--version` не совпал с `PROJECT_STATE.json` | пересоберите с явными `-DLLAMA_BUILD_COMMIT` и `-DLLAMA_BUILD_NUMBER` |
| `pipeline\.venv` не создаётся | проверьте `python --version` ≥ 3.11 и права на запись |
| `doctor`: GPU не виден | `nvidia-smi`; без GPU используйте `-NoCuda`, но прогоны будут медленными |

После исправлений повторите:

```powershell
powershell -File scripts\run.ps1 -Stage doctor -Profile standard
& .\pipeline\.venv\Scripts\python.exe scripts\check_state.py
```
