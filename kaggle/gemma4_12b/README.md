# Gemma 4 12B: прогон на Kaggle GPU

Модель в 6 ГБ локальной RTX 3050 не влезает, поэтому прогон идёт в ноутбуке Kaggle
(`odinodinodin/hs-gemma4-12b`, приватный, `machine_shape: NvidiaTeslaT4`).
Всё делается в одной GPU-сессии, потому что `/kaggle/working` умирает вместе с ней.

## Что делает ноутбук (7 ячеек, `make_nb.py`)

| ячейка | что |
|---|---|
| `setup` | `nvidia-smi`, `nproc`, `df`; находит каталог датасета `hs-src` в `/kaggle/input` по имени `hidden-states.cpp`; задаёт пути (`/tmp/llama.cpp`, `/tmp/models`) |
| `build` | клонирует пинованный `d4cff114c`, кладёт наши 3 файла, собирает `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75`; при падении configure — повтор с `-DGGML_CUDA_NO_VMM=ON` |
| `download` | `unsloth/gemma-4-12b-it-GGUF`, файл `gemma-4-12b-it-Q4_K_M.gguf` → `/tmp/models` (в `working` не пишется, иначе не хватит 20 ГБ) |
| `pilot` | первые 40 строк `pilot.jsonl`, те же флаги, что у остальных моделей: `--generate --max-new-tokens 64 --max-len 1024 --layers all -ngl 99` |
| `gate` | разбирает `done: N/N ok=… skipped=… ex/s` и `shape=[…]` |
| `main` | полный прогон 6000 строк, только если гейт прошёл |
| `final` | `run_summary.json`: файлы с размерами, строки meta, `status_ok`, `truncated`, `ids_unique`, поля `config.json` |

**Гейт:** `ok == total`, `skipped == 0`, `ex/s > 0.2` и оценка `6000 / ex/s < 4 ч`.
Не прошёл — 6000 строк не запускаются, чтобы не жечь квоту.

Сборка и модель живут в `/tmp`, в `/kaggle/working` попадают только `out/pilot_gemma4_12b`,
`out/main_gemma4_12b`, логи и `run_summary.json`.

## Запуск и выгрузка

```powershell
powershell -File kaggle\gemma4_12b\push.ps1              # датасет hs-src из файлов проекта + ядро
powershell -File kaggle\gemma4_12b\push.ps1 -SkipDataset # вход не менялся
powershell -File kaggle\watch.ps1 -IntervalSec 60        # ждать финиша, снять лог и квоту

$KAG = 'C:\Windows\Temp\opencode\kgvenv\Scripts\kaggle.exe'
& $KAG kernels output odinodinodin/hs-gemma4-12b -p kaggle\gemma4_12b\out
Copy-Item kaggle\gemma4_12b\out\out\main_gemma4_12b\* pipeline\out\main_gemma4_12b\
```

Дальше — обычный конвейер из `pipeline\`: `verify_run.py --full` → `label.py` →
`baseline.py` → `probe.py`.

## Грабли (проверено на четырёх версиях ядра)

- датасет монтируется в `/kaggle/input/datasets/<owner>/<dataset>`, поэтому файлы ищутся
  `os.walk`, а не по жёсткому пути;
- структура датасета обязана быть плоской (`kaggle datasets create` без `--dir-mode`
  пропускает подкаталоги);
- `/usr/local/cuda` только на чтение → stub-каталог `/tmp/cuda_stubs` с симлинком на
  найденный `libcuda.so.1` и `-DCMAKE_LIBRARY_PATH`;
- `kaggle kernels logs` для живого ядра пуст — статус смотреть через
  `kernels status` или в браузере;
- квота: 30 ч GPU в неделю, расход записывается в таблицу `AGENTS.md`.
