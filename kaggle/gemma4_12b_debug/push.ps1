# Пушит debug-ядро odinodinodin/hs-gemma4-12b-debug (актуальный llama.cpp, без датасета).
# Запуск: powershell -File kaggle\gemma4_12b_debug\push.ps1
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$root = (Resolve-Path (Join-Path $here '..\..')).Path

$KAG = $env:KAGGLE_EXE
if (-not $KAG) {
    $KAG = 'C:\Windows\Temp\opencode\kgvenv\Scripts\kaggle.exe'
    if (-not (Test-Path $KAG)) { $KAG = 'kaggle' }
}

$env:KAGGLE_API_TOKEN = (Get-Content (Join-Path $root '.env') |
    Select-String -Pattern '^KAGGLE_API_TOKEN=').Line.Substring(17).Trim()

Write-Host '--- quota before ---'
& $KAG quota 2>&1

& python (Join-Path $here 'make_nb.py')
if ($LASTEXITCODE -ne 0) { throw 'make_nb.py failed' }

& $KAG kernels push -p (Join-Path $here '_build\kernel') --accelerator NvidiaTeslaT4 2>&1 |
    Select-Object -Last 3