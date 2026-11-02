# Gemma 4 12B: прогон на Kaggle GPU

Модель в 6 ГБ локальной RTX 3050 не влезает, поэтому прогон идёт в ноутбуке Kaggle
(`odinodinodin/hs-gemma4-12b`, приватный, `machine_shape: NvidiaTeslaT4`).
Всё делается в одной GPU-сессии, потому что `/kaggle/working` умирает вместе с ней.

## Что делает ноутбук (8 ячеек, `make_nb.py`)

| ячейка | что |
|---|---|
| `setup` | `nvidia-smi`, `nproc`, `df`; находит каталог датасета `hs-src` в `/kaggle/input` по имени `hidden-states.cpp`; задаёт пути (`/tmp/llama.cpp`, `/tmp/models`) |
| `build` | клонирует пинованный `d4cff114c`, кладёт наши 3 файла, собирает `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75`; при падении configure — повтор с `-DGGML_CUDA_NO_VMM=ON` |
| `download` | `unsloth/gemma-4-12b-it-GGUF`, файл `gemma-4-12b-it-Q4_K_M.gguf` → `/tmp/models` (в `working` не пишется, иначе не хватит 20 ГБ) |
| `easy` | 8 простых вопросов; проверяет ответы до дорогого пилота и полного прогона |
| `pilot` | первые 40 строк `pilot.jsonl`, те же флаги, что у остальных моделей: `--generate --max-new-tokens 64 --max-len 1024 --layers all -ngl 99` |
| `gate` | проверяет easy8, `ok/skipped`, скорость, мета-текст и утечки `<|channel>` |
| `main` | полный прогон 6000 строк, только если гейт прошёл |
| `final` | `run_summary.json`: файлы с размерами, строки meta, `status_ok`, `truncated`, `ids_unique`, поля `config.json` |

**Гейт:** easy8 ≥ 7/8 (ожидается 8/8), `ok == total`, `skipped == 0`, 0 мета-текста,
0 утечек `<|channel>`, `ex/s > 0.2` и оценка `6000 / ex/s < 4 ч`.
Не прошёл — 6000 строк не запускаются, чтобы не жечь квоту.

Сборка и модель живут в `/tmp`, в `/kaggle/working` попадают только easy8,
`out/pilot_gemma4_12b`, `out/main_gemma4_12b_v3`, логи и `run_summary.json`.

## Запуск и выгрузка

```powershell
powershell -File kaggle\gemma4_12b\push.ps1              # датасет hs-src из файлов проекта + ядро
powershell -File kaggle\gemma4_12b\push.ps1 -SkipDataset # вход не менялся
powershell -File kaggle\watch.ps1 -IntervalSec 60        # ждать финиша, снять лог и квоту

$KAG = 'C:\Windows\Temp\opencode\kgvenv\Scripts\kaggle.exe'
& $KAG kernels output odinodinodin/hs-gemma4-12b -p kaggle\gemma4_12b\out\v5
Copy-Item kaggle\gemma4_12b\out\v5\out\main_gemma4_12b_v3\* pipeline\out\main_gemma4_12b_v3\
```

Выполнены `verify_run.py --full --input data/main.jsonl` → `label.py` → `baseline.py`.
Проба 12B отложена по решению пользователя.

`main_gemma4_12b` без суффикса — непригодный старый прогон без ведущего `<bos>`
(`report.md` §22); не перезаписывать и не использовать.

## Результат v5 (2026-11-02)

- easy8 7/8, pilot40 40/40 без обрезок и мета-текста;
- main 6000/6000, 21 truncated, 16 retokenize mismatch;
- 2012.1 s (2.98 ex/s), `[6000,48,3840]`, по 2109.4 MiB;
- `filtered_b` 0.572, baseline test 0.813/0.826;
- сессия 61.1 мин, расход 1.01 GPU-ч; всего использовано 5.99/30 ч.

## Грабли

- датасет монтируется в `/kaggle/input/datasets/<owner>/<dataset>`, поэтому файлы ищутся
  `os.walk`, а не по жёсткому пути;
- структура датасета обязана быть плоской (`kaggle datasets create` без `--dir-mode`
  пропускает подкаталоги);
- `/usr/local/cuda` только на чтение → stub-каталог `/tmp/cuda_stubs` с симлинком на
  найденный `libcuda.so.1` и `-DCMAKE_LIBRARY_PATH`;
- `kaggle kernels logs` для живого ядра пуст — статус смотреть через
  `kernels status` или в браузере;
- `Set-Content -Encoding utf8` в Windows PowerShell добавляет BOM в
  `dataset-metadata.json`; писать через `UTF8Encoding(false)`;
- Kaggle CLI пишет progress в stderr даже при rc=0; `push.ps1` временно переводит
  `$ErrorActionPreference` в `Continue` и проверяет код возврата сам;
- квота: 30 ч GPU в неделю, расход записывается в таблицу `AGENTS.md`.
