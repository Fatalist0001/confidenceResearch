# Kaggle: прогоны на GPU вне этой машины

Каждый подкаталог — один кернел Kaggle: генератор ноутбука `make_nb.py`, скрипт
`push.ps1` (собрать датасет + запушить ядро) и `README.md` с описанием прогона.

```
powershell -File kaggle\gemma4_12b\push.ps1            # датасет + ядро
powershell -File kaggle\gemma4_12b\push.ps1 -SkipDataset  # только ядро (вход не менялся)
```

Собранное и скачанное живёт рядом, но в git не попадает (`.gitignore`):
`_build\` — датасет и ноутбук, `out\` — выгрузка `kaggle kernels output`.

Токен `KAGGLE_API_TOKEN` берётся из `.env` в корне репозитория. CLI — `kaggle`
из PATH либо `KAGGLE_EXE`; на этой машине venv лежит в
`C:\Windows\Temp\opencode\kgvenv`.

Забрать результат прогона и разложить в пайплайн (пример для 12B):

```powershell
$KAG = 'C:\Windows\Temp\opencode\kgvenv\Scripts\kaggle.exe'
& $KAG kernels output odinodinodin/hs-gemma4-12b -p kaggle\gemma4_12b\out
Copy-Item kaggle\gemma4_12b\out\out\main_gemma4_12b\* pipeline\out\main_gemma4_12b\
```

Квоты и правила (пилот перед полным прогоном, запись расхода) — в `AGENTS.md`,
раздел «Kaggle: GPU вне этой машины».
