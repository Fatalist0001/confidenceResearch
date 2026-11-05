param(
    [switch]$SkipDataset,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$root = (Resolve-Path (Join-Path $here '..\..')).Path
$ll = Join-Path $root 'llama.cpp-hiddenstates-new'
$data = Join-Path $root 'pipeline\data'
$kaggle = $env:KAGGLE_EXE
if (-not $kaggle) {
    $kaggle = 'C:\Windows\Temp\opencode\kgvenv\Scripts\kaggle.exe'
    if (-not (Test-Path $kaggle)) { $kaggle = 'kaggle' }
}

$tokenLine = Get-Content (Join-Path $root '.env') | Select-String '^KAGGLE_API_TOKEN=' | Select-Object -First 1
if (-not $tokenLine) { throw 'KAGGLE_API_TOKEN is missing from .env' }
$env:KAGGLE_API_TOKEN = ($tokenLine.Line -replace '^KAGGLE_API_TOKEN=', '').Trim()

function Invoke-Kaggle {
    param([string[]]$Arguments)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $output = @(& $kaggle @Arguments 2>&1)
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldPreference
    $output | ForEach-Object { Write-Host $_ }
    if ($exitCode -ne 0) { throw "kaggle command failed ($exitCode): $($Arguments -join ' ')" }
}

if (-not $DryRun) {
    Write-Host '=== quota before ==='
    Invoke-Kaggle @('quota')
}

$build = Join-Path $here '_build'
$dataset = Join-Path $build 'ds'
if (Test-Path -LiteralPath $dataset) { Remove-Item -LiteralPath $dataset -Recurse -Force }
New-Item -ItemType Directory -Force -Path $dataset | Out-Null

$sources = [ordered]@{
    'hidden-states.cpp'            = Join-Path $ll 'examples\hidden-states\hidden-states.cpp'
    'hidden-states-CMakeLists.txt' = Join-Path $ll 'examples\hidden-states\CMakeLists.txt'
    'prompt_dump.cpp'              = Join-Path $ll 'examples\hidden-states\prompt_dump.cpp'
    'examples-CMakeLists.txt'      = Join-Path $ll 'examples\CMakeLists.txt'
    'pilot.jsonl'                  = Join-Path $data 'pilot.jsonl'
    'main.jsonl'                   = Join-Path $data 'main.jsonl'
}

$manifestFiles = [ordered]@{}
foreach ($entry in $sources.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $entry.Value)) { throw "missing source: $($entry.Value)" }
    $destination = Join-Path $dataset $entry.Key
    Copy-Item -LiteralPath $entry.Value -Destination $destination -Force
    $manifestFiles[$entry.Key] = [ordered]@{
        bytes = (Get-Item -LiteralPath $destination).Length
        sha256 = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$manifest = [ordered]@{
    schema_version = 1
    created_utc = (Get-Date).ToUniversalTime().ToString('o')
    files = $manifestFiles
}
[IO.File]::WriteAllText(
    (Join-Path $dataset 'source-manifest.json'),
    ($manifest | ConvertTo-Json -Depth 8),
    [Text.UTF8Encoding]::new($false))

$datasetMetadata = [ordered]@{
    title = 'hs source and data'
    id = 'odinodinodin/hs-src'
    licenses = @(@{ name = 'other' })
}
[IO.File]::WriteAllText(
    (Join-Path $dataset 'dataset-metadata.json'),
    ($datasetMetadata | ConvertTo-Json -Depth 5),
    [Text.UTF8Encoding]::new($false))

if ($DryRun) {
    Write-Host 'dataset upload skipped in dry run'
} elseif (-not $SkipDataset) {
    Write-Host '=== upload dataset version ==='
    Invoke-Kaggle @('datasets', 'version', '-p', $dataset, '-m', "add 26B prompt preflight $(Get-Date -Format 'yyyy-MM-dd HH:mm')")
} else {
    Write-Host 'dataset upload skipped by request'
}

& python (Join-Path $here 'make_nb.py')
if ($LASTEXITCODE -ne 0) { throw 'make_nb.py failed' }

$kernelDir = Join-Path $build 'kernel'
$metadata = Get-Content (Join-Path $kernelDir 'kernel-metadata.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($metadata.id -ne 'odinodinodin/hs-gemma4-26b-a4b') { throw "wrong kernel id: $($metadata.id)" }
if ($metadata.machine_shape -ne 'NvidiaTeslaT4' -or -not $metadata.enable_gpu) { throw 'wrong accelerator metadata' }

if ($DryRun) {
    Write-Host 'dry run passed: dataset and kernel payloads were built; no Kaggle API mutation was made'
    return
}

Write-Host '=== push kernel ==='
Invoke-Kaggle @('kernels', 'push', '-p', $kernelDir, '--accelerator', 'NvidiaTeslaT4')

Write-Host '=== status ==='
$statusOk = $false
for ($attempt = 1; $attempt -le 6; $attempt++) {
    try {
        Invoke-Kaggle @('kernels', 'status', 'odinodinodin/hs-gemma4-26b-a4b')
        $statusOk = $true
        break
    } catch {
        if ($attempt -eq 6) { throw }
        Write-Host "status is not visible yet (attempt $attempt/6), retrying in 5 s"
        Start-Sleep -Seconds 5
    }
}
if (-not $statusOk) { throw 'kernel status could not be confirmed' }
Write-Host '=== quota after submission ==='
Invoke-Kaggle @('quota')
