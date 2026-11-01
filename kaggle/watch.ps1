# Ждёт завершения ядра Kaggle и складывает лог рядом.
# Запуск: powershell -File kaggle\watch.ps1
#         powershell -File kaggle\watch.ps1 -Kernel odinodinodin/hs-gemma4-12b -IntervalSec 30
param(
    [string]$Kernel = 'odinodinodin/hs-gemma4-12b',
    [int]$IntervalSec = 60,
    [string]$LogDir = ''
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

$KAG = $env:KAGGLE_EXE
if (-not $KAG) {
    $KAG = 'C:\Windows\Temp\opencode\kgvenv\Scripts\kaggle.exe'
    if (-not (Test-Path $KAG)) { $KAG = 'kaggle' }
}
$env:KAGGLE_API_TOKEN = (Get-Content (Join-Path $root '.env') |
    Select-String -Pattern '^KAGGLE_API_TOKEN=').Line.Substring(17).Trim()

if (-not $LogDir) { $LogDir = Join-Path $PSScriptRoot 'gemma4_12b\out' }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stamp = $Kernel.Split('/')[-1]
$statusLog = Join-Path $LogDir "$stamp.status.log"
$fullLog   = Join-Path $LogDir "$stamp.kernel.log"

function Say($msg) {
    $line = "$(Get-Date -Format 'HH:mm:ss') $msg"
    $line | Tee-Object -FilePath $statusLog -Append | Out-Host
}

Say "watch $Kernel, interval ${IntervalSec}s, log $statusLog"
$t0 = Get-Date
while ($true) {
    $out = & $KAG kernels status $Kernel 2>&1 | Select-Object -Last 1
    Say ($out -join ' ')
    if ($out -notmatch 'RUNNING|QUEUED|PENDING|STARTING') { break }
    Start-Sleep -Seconds $IntervalSec
}

$mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
Say "finished after $mins min"

# полный лог ядра и остаток квоты
& $KAG kernels logs $Kernel 2>&1 | Out-File -Encoding utf8 $fullLog
Say "kernel log -> $fullLog"
& $KAG quota 2>&1 | Out-Host
