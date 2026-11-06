# Prepare the fixed-budget LFM2.5 Thinking input consumed by scripts/run.ps1.

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('pilot', 'main')]
    [string]$Stage,

    [Parameter(Mandatory = $true)]
    [string]$Model,

    [string]$Tag = 'lfm25_1p2b_thinking_r32_q4_v2',

    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Pipeline = Join-Path $Root 'pipeline'
$Out = Join-Path $Pipeline 'out'
$Data = Join-Path $Pipeline 'data'
$Python = Join-Path $Pipeline '.venv\Scripts\python.exe'
$State = Get-Content (Join-Path $Root 'PROJECT_STATE.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$Binary = Join-Path $Root $State.binaries.active.exe
$ReasoningBudget = 32
$FinalBudget = 64

if (-not (Test-Path -LiteralPath $Model)) { throw "model not found: $Model" }
if (-not (Test-Path -LiteralPath $Binary)) { throw "binary not found: $Binary" }
if (-not (Test-Path -LiteralPath $Python)) { throw "python venv not found: $Python" }

$SourceName = if ($Stage -eq 'pilot') { 'pilot.jsonl' } else { 'main.jsonl' }
$Source = Join-Path $Data $SourceName
$ExpectedRows = if ($Stage -eq 'pilot') { 200 } else { 6000 }
$Stem = "lfm25_thinking_${Stage}_${Tag}"
$ReasoningDir = Join-Path $Out "${Stem}_reasoning"
$ReasoningLog = "$ReasoningDir.log"
$ProtocolDir = Join-Path $Out "${Stem}_protocol"
$FinalPrompts = Join-Path $ProtocolDir 'final_prompts.jsonl'
$FinalDir = Join-Path $Out "${Stem}_finalgen"
$FinalLog = "$FinalDir.log"
$StateInput = Join-Path $Data "${Stage}_${Tag}_final.jsonl"
$EvalMeta = Join-Path $ProtocolDir 'eval_meta.jsonl'
$Summary = Join-Path $ProtocolDir 'summary.json'
$Preparation = Join-Path $ProtocolDir 'preparation.json'
$ReasoningMeta = Join-Path $ReasoningDir 'meta.jsonl'

if ($Resume) {
    foreach ($path in @($ProtocolDir, $FinalDir, $FinalLog, $StateInput)) {
        if (Test-Path -LiteralPath $path) { throw "refusing to overwrite $path" }
    }
    if (-not (Test-Path -LiteralPath $ReasoningMeta)) {
        throw "nothing to resume: $ReasoningMeta does not exist"
    }
} else {
    foreach ($path in @($ReasoningDir, $ReasoningLog, $ProtocolDir, $FinalDir, $FinalLog, $StateInput)) {
        if (Test-Path -LiteralPath $path) { throw "refusing to overwrite $path" }
    }
}

if ($Stage -eq 'main') {
    $PilotGate = Join-Path $Out "pilot_$Tag\gate.json"
    if (-not (Test-Path -LiteralPath $PilotGate)) { throw "main preparation requires passed pilot gate: $PilotGate" }
    $Gate = Get-Content $PilotGate -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $Gate.passed) { throw "pilot gate did not pass: $PilotGate" }
    $PilotPreparation = Join-Path $Out "lfm25_thinking_pilot_${Tag}_protocol\preparation.json"
    if (-not (Test-Path -LiteralPath $PilotPreparation)) { throw "pilot preparation record is missing: $PilotPreparation" }
    $PilotTiming = Get-Content $PilotPreparation -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([double]$PilotTiming.estimated_full_hours -gt 6.0) {
        throw "estimated full preparation exceeds 6 hours: $($PilotTiming.estimated_full_hours)"
    }
}

function Quote-Arg([string]$Value) {
    if ($Value -match '[\s"]') { return '"' + ($Value -replace '"', '\"') + '"' }
    return $Value
}

function Invoke-Logged([string[]]$Argv, [string]$LogPath) {
    if (Test-Path -LiteralPath $LogPath) { throw "refusing to overwrite $LogPath" }
    $ArgLine = ($Argv | ForEach-Object { Quote-Arg $_ }) -join ' '
    $Stdout = "$LogPath.stdout"
    $Process = Start-Process -FilePath $Binary -ArgumentList $ArgLine -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $Stdout -RedirectStandardError $LogPath
    if (Test-Path -LiteralPath $Stdout) {
        Get-Content -LiteralPath $Stdout -Encoding UTF8 | Add-Content -LiteralPath $LogPath -Encoding UTF8
        Remove-Item -LiteralPath $Stdout
    }
    if ($Process.ExitCode -ne 0) { throw "binary exited with $($Process.ExitCode): $LogPath" }
}

& $Python (Join-Path $Pipeline 'preflight.py') --input $Source --expect-rows $ExpectedRows --require-instruction
if ($LASTEXITCODE -ne 0) { throw 'source preflight failed' }

$ResumeFromRows = 0
$RestRows = 0
$ReasoningTotalSeconds = 0.0
$ReasoningSeconds = 0.0

if ($Resume) {
    $MetaCount = & $Python (Join-Path $Pipeline 'lfm_reasoning.py') meta-count --meta $ReasoningMeta | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'could not count partial reasoning meta' }
    $ResumeFromRows = [int]$MetaCount.rows
    if ($ResumeFromRows -lt 1 -or $ResumeFromRows -ge $ExpectedRows) {
        throw "partial reasoning meta has $ResumeFromRows of $ExpectedRows rows; nothing to resume"
    }
    New-Item -ItemType Directory -Path $ProtocolDir | Out-Null
    $RestInput = Join-Path $ProtocolDir "reasoning_input_from_$ResumeFromRows.jsonl"
    & $Python (Join-Path $Pipeline 'lfm_reasoning.py') make-remaining `
        --in $Source --out $RestInput --start $ResumeFromRows
    if ($LASTEXITCODE -ne 0) { throw 'could not build the remaining reasoning input' }
    $RestRows = $ExpectedRows - $ResumeFromRows
    $RestDir = Join-Path $Out "${Stem}_reasoning_rest_from_$ResumeFromRows"
    $RestLog = "$RestDir.log"
    if ((Test-Path -LiteralPath $RestDir) -or (Test-Path -LiteralPath $RestLog)) {
        throw "refusing to overwrite $RestDir or $RestLog"
    }

    Write-Host "== reasoning pass: resume from row $ResumeFromRows, $RestRows rows"
    $ReasoningStart = Get-Date
    Invoke-Logged @(
        '-m', $Model, '--batch', $RestInput, '--out', $RestDir, '--generate',
        '--max-new-tokens', [string]$ReasoningBudget, '--max-len', '1024', '--layers', 'all', '-ngl', '99'
    ) $RestLog
    $ReasoningSeconds = ((Get-Date) - $ReasoningStart).TotalSeconds

    $MergedReasoning = Join-Path $ProtocolDir 'reasoning_meta.jsonl'
    & $Python (Join-Path $Pipeline 'lfm_reasoning.py') merge-reasoning --questions $Source `
        --part $ReasoningMeta --part (Join-Path $RestDir 'meta.jsonl') --out $MergedReasoning
    if ($LASTEXITCODE -ne 0) { throw 'could not merge reasoning meta' }
    $ReasoningMeta = $MergedReasoning
    $ReasoningTotalSeconds = if ($RestRows -gt 0) { $ReasoningSeconds * $ExpectedRows / $RestRows } else { 0.0 }
} else {
    Write-Host "== reasoning pass: $Stage, budget $ReasoningBudget"
    $ReasoningStart = Get-Date
    Invoke-Logged @(
        '-m', $Model, '--batch', $Source, '--out', $ReasoningDir, '--generate',
        '--max-new-tokens', [string]$ReasoningBudget, '--max-len', '1024', '--layers', 'all', '-ngl', '99'
    ) $ReasoningLog
    $ReasoningSeconds = ((Get-Date) - $ReasoningStart).TotalSeconds
    $ReasoningTotalSeconds = $ReasoningSeconds
    New-Item -ItemType Directory -Path $ProtocolDir | Out-Null
}

& $Python (Join-Path $Pipeline 'lfm_reasoning.py') build-final-prompts `
    --questions $Source --reasoning-meta $ReasoningMeta `
    --out $FinalPrompts --budget $ReasoningBudget
if ($LASTEXITCODE -ne 0) { throw 'final prompt build failed' }

Write-Host "== final pass: $Stage, budget $FinalBudget"
$FinalStart = Get-Date
Invoke-Logged @(
    '-m', $Model, '--batch', $FinalPrompts, '--out', $FinalDir, '--no-template', '--generate',
    '--max-new-tokens', [string]$FinalBudget, '--max-len', '1024', '--layers', 'all', '-ngl', '99'
) $FinalLog
$FinalSeconds = ((Get-Date) - $FinalStart).TotalSeconds

& $Python (Join-Path $Pipeline 'lfm_reasoning.py') extract-finals `
    --prompts $FinalPrompts --final-meta (Join-Path $FinalDir 'meta.jsonl') `
    --state-input $StateInput --eval-meta $EvalMeta --summary $Summary
if ($LASTEXITCODE -ne 0) { throw 'final extraction failed' }

$ProtocolSummary = Get-Content $Summary -Raw -Encoding UTF8 | ConvertFrom-Json
if ([double]$ProtocolSummary.coverage -lt 0.95) {
    throw "final extraction coverage is below 0.95: $($ProtocolSummary.coverage)"
}

$Scale = 6000.0 / $ExpectedRows
$ReasoningRowsGenerated = if ($Resume) { $RestRows } else { $ExpectedRows }
$Timing = [ordered]@{
    stage                    = $Stage
    tag                      = $Tag
    rows                     = $ExpectedRows
    reasoning_budget         = $ReasoningBudget
    final_budget             = $FinalBudget
    resumed_from_rows        = $ResumeFromRows
    reasoning_rows_generated = $ReasoningRowsGenerated
    reasoning_seconds        = [math]::Round($ReasoningSeconds, 3)
    estimated_reasoning_seconds = [math]::Round($ReasoningTotalSeconds, 3)
    final_seconds            = [math]::Round($FinalSeconds, 3)
    estimated_full_hours     = [math]::Round(($ReasoningTotalSeconds + $FinalSeconds) * $Scale / 3600.0, 3)
    state_input              = $StateInput
    coverage                 = [double]$ProtocolSummary.coverage
}
($Timing | ConvertTo-Json -Depth 4) | Out-File -FilePath $Preparation -Encoding utf8

Write-Host "prepared: $StateInput"
Write-Host "coverage: $($ProtocolSummary.extracted)/$($ProtocolSummary.rows)"
Write-Host "estimated full preparation: $($Timing.estimated_full_hours) h"
