# Собирает датасет odinodinodin/hs-src из файлов проекта и пушит ядро Kaggle.
# Запуск: powershell -File kaggle\gemma4_12b\push.ps1 [-SkipDataset]
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

function Invoke-KaggleCommand {
    param([string[]]$Arguments)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $output = @(& $KAG @Arguments 2>&1)
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldPreference
    return [pscustomobject]@{ Output = $output; ExitCode = $exitCode }
}

$build = Join-Path $here '_build'
$ds    = Join-Path $build 'ds'
New-Item -ItemType Directory -Force -Path $ds | Out-Null

# --- датасет: плоская структура (kaggle не умеет подкаталоги без --dir-mode) ---
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
$dsMetaJson = $dsMeta | ConvertTo-Json -Depth 5
[IO.File]::WriteAllText(
    (Join-Path $ds 'dataset-metadata.json'),
    $dsMetaJson,
    [Text.UTF8Encoding]::new($false))

if (-not $SkipDataset) {
    $result = Invoke-KaggleCommand @('datasets', 'version', '-p', $ds, '-m', "hs sources $(Get-Date -Format 'yyyy-MM-dd HH:mm')")
    if ($result.ExitCode -ne 0) {
        Write-Host 'datasets version failed, trying create'
        $result = Invoke-KaggleCommand @('datasets', 'create', '-p', $ds)
        if ($result.ExitCode -ne 0) {
            $result.Output | Select-Object -Last 10
            throw 'dataset upload failed'
        }
    }
    Write-Host 'dataset pushed'
}

# --- ноутбук + метаданные ядра ---
& python (Join-Path $here 'make_nb.py')
if ($LASTEXITCODE -ne 0) { throw 'make_nb.py failed' }

$result = Invoke-KaggleCommand @('kernels', 'push', '-p', (Join-Path $build 'kernel'), '--accelerator', 'NvidiaTeslaT4')
$result.Output | Select-Object -Last 2
if ($result.ExitCode -ne 0) { throw 'kernel push failed' }

$result = Invoke-KaggleCommand @('quota')
$result.Output
if ($result.ExitCode -ne 0) { throw 'quota failed' }
