# Kaggle: прогоны на GPU вне этой машины

Каждый подкаталог — один кернел Kaggle: генератор ноутбука `make_nb.py`, скрипт
`push.ps1` (собрать датасет + запушить ядро) и `README.md` с описанием прогона.

```
powershell -File kaggle\gemma4_12b\push.ps1            # датасет + ядро
powershell -File kaggle\gemma4_12b\push.ps1 -SkipDataset  # только ядро (вход не менялся)
powershell -File kaggle\gemma4_26b_a4b\push.ps1        # 26B-A4B, 2×T4
```

Собранное и скачанное живёт рядом, но в git не попадает (`.gitignore`):
`_build\` — датасет и ноутбук, `out\` — выгрузка `kaggle kernels output`.

Токен `KAGGLE_API_TOKEN` берётся из `.env` в корне репозитория. CLI — `kaggle`
из PATH либо `KAGGLE_EXE`; на этой машине venv лежит в
`C:\Windows\Temp\opencode\kgvenv`.

Забрать результат прогона и разложить в пайплайн (пример для 12B):

```powershell
$KAG = 'C:\Windows\Temp\opencode\kgvenv\Scripts\kaggle.exe'
& $KAG kernels output odinodinodin/hs-gemma4-12b -p kaggle\gemma4_12b\out\v5
Copy-Item kaggle\gemma4_12b\out\v5\out\main_gemma4_12b_v3\* pipeline\out\main_gemma4_12b_v3\
```

`main_gemma4_12b` без суффикса — непригодный прогон без ведущего `<bos>`; его не
перезаписывать и не использовать. Валидный `main_gemma4_12b_v3` завершён 2026-11-02;
проба посчитана локально 2026-11-04 (test AUROC 0.9107), остальные артефакты лежат
в `pipeline/out/main_gemma4_12b_v3`.

Gemma 4 26B-A4B:

```powershell
& $KAG kernels output odinodinodin/hs-gemma4-26b-a4b `
  -p kaggle\gemma4_26b_a4b\out\v2-retrieved-20261105
Copy-Item kaggle\gemma4_26b_a4b\out\v2-retrieved-20261105\out\main_gemma4_26b_a4b_udq4km\* `
  pipeline\out\main_gemma4_26b_a4b_udq4km\
```

Валидный v2 завершён 2026-11-04 и выгружен 2026-11-05: 6000/6000, 32 truncated,
форма `[6000,30,2816]`; локальная probe C/16 дала test AUROC 0.9077. v1 остановился
до пилота из-за опечатки и сохранён только как свидетельство сбоя.

Квоты и правила (пилот перед полным прогоном, запись расхода) — в `AGENTS.md`,
раздел «Kaggle: GPU вне этой машины».

Датасет с нашими исходниками `odinodinodin/hs-src` **общий** для всех ядер: `push.ps1`
любого ядра кладёт туда `hidden-states.cpp` и два `CMakeLists.txt` из
`llama.cpp-hiddenstates\`, поэтому правка кода уезжает сразу во все ядра. Если менялся
только код, а входы (`pilot.jsonl`, `main.jsonl`) нет — пушить можно с `-SkipDataset`,
но тогда датасет надо было обновить предыдущим пушем.

Гейт ядра обязан проверять **ответы**, а не только счётчики: у `hs-gemma4-12b` v1 гейт
прошёл при `ok=6000` и одновременно мусорных ответах (strict 9/3702). В текущих ядрах
для этого есть easy8 (порог 7/8) и счётчики мета-текста/channel leak в пилоте.
