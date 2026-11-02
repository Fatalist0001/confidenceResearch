# Обновляет датасет odinodinodin/hs-src (в нём лежит наш портированный
# hidden-states.cpp) и пушит ядро odinodinodin/hs-gemma4-12b-master-gate.
# Запуск: powershell -File kaggle\gemma4_12b_hs_master\push.ps1 [-SkipDataset]
param(
    [switch]$SkipDataset
)

$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$root = (Resolve-Path (Join-Path $here '..\..')).Path
$LL   = Join-Path $root 'llama.cpp-hiddenstates'
$data = Join-Path $root 'pipeline\data'

$KAG = $env:KAGGLE_EXE
if (-not $KAG) {
    $KAG = 'C:\Windows\Temp\opencode\kgvenv\Scripts\kaggle.exe'
    if (-not (Test-Path $KAG)) { $KAG = 'kaggle' }
}
$env:KAGGLE_API_TOKEN = (Get-Content (Join-Path $root '.env') |
    Select-String -Pattern '^KAGGLE_API_TOKEN=').Line.Substring(17).Trim()

Write-Host '--- quota before ---'
& $KAG quota 2>&1

$build = Join-Path $here '_build'
$ds    = Join-Path $build 'ds'
New-Item -ItemType Directory -Force -Path $ds | Out-Null

Copy-Item (Join-Path $LL 'examples\hidden-states\hidden-states.cpp') (Join-Path $ds 'hidden-states.cpp') -Force
Copy-Item (Join-Path $LL 'examples\hidden-states\CMakeLists.txt')    (Join-Path $ds 'hidden-states-CMakeLists.txt') -Force
Copy-Item (Join-Path $LL 'examples\CMakeLists.txt')                  (Join-Path $ds 'examples-CMakeLists.txt') -Force
Copy-Item (Join-Path $data 'pilot.jsonl') (Join-Path $ds 'pilot.jsonl') -Force
Copy-Item (Join-Path $data 'main.jsonl')  (Join-Path $ds 'main.jsonl')  -Force

$dsMeta = @{
    title    = 'hs source and data'
    id       = 'odinodinodin/hs-src'
    licenses = @(@{ name = 'other' })
}
$dsMeta | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $ds 'dataset-metadata.json') -Encoding utf8
# kaggle CLI 2.2.5 не читает BOM: перезаписываем без него
$metaPath = Join-Path $ds 'dataset-metadata.json'
[IO.File]::WriteAllText($metaPath, (Get-Content $metaPath -Raw),
    (New-Object System.Text.UTF8Encoding($false)))

if (-not $SkipDataset) {
    # PS 5 считает stderr нативной команды ошибкой при -Stop: на время вызовов CLI ослабляем
    $ErrorActionPreference = 'Continue'
    & $KAG datasets version -p $ds -m "hs sources: master port $(Get-Date -Format 'yyyy-MM-dd HH:mm')" 2>&1 |
        Select-Object -Last 2
    $rc = $LASTEXITCODE
    if ($rc -ne 0) {
        Write-Host 'datasets version failed, trying create'
        & $KAG datasets create -p $ds 2>&1 | Select-Object -Last 2
        $rc = $LASTEXITCODE
    }
    $ErrorActionPreference = 'Stop'
    if ($rc -ne 0) { throw 'dataset upload failed' }
    Write-Host 'dataset pushed'
}

& python (Join-Path $here 'make_nb.py')
if ($LASTEXITCODE -ne 0) { throw 'make_nb.py failed' }

& $KAG kernels push -p (Join-Path $build 'kernel') --accelerator NvidiaTeslaT4 2>&1 |
    Select-Object -Last 3