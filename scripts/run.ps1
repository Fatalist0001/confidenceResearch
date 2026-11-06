# Stage runner for llama-hidden-states.
#
# It replaces "copy the commands from AGENTS.md and remember the exceptions" with a
# fixed sequence that refuses unsafe transitions:
#   * never writes into an existing run directory;
#   * never starts generation without a matching prompt/input provenance record;
#   * never starts 6000 rows without a passed pilot gate;
#   * never accepts a pilot until labels contain both classes;
#   * never runs probe.py before verify_run.py passed;
#   * never runs a deferred experiment without -ConfirmDeferred;
#   * never passes -b/-ub (batch mode derives them from --max-len).
#
# Examples:
#   powershell -File scripts\run.ps1 -Stage doctor -Profile standard -Model D:\models\x.gguf
#   powershell -File scripts\run.ps1 -Stage inputs -Profile qwen -Tag qwen4b_v2
#   powershell -File scripts\run.ps1 -Stage prompt -Profile qwen -Tag qwen4b_v2 -Model D:\models\x.gguf
#   powershell -File scripts\run.ps1 -Stage pilot  -Profile qwen -Tag qwen4b_v2 -Model D:\models\x.gguf
#   powershell -File scripts\run.ps1 -Stage main   -Profile qwen -Tag qwen4b_v2 -Model D:\models\x.gguf
#   powershell -File scripts\run.ps1 -Stage label  -Profile qwen -Tag qwen4b_v2
#   powershell -File scripts\run.ps1 -Stage verify -Profile qwen -Tag qwen4b_v2
#   powershell -File scripts\run.ps1 -Stage baseline -Profile qwen -Tag qwen4b_v2
#   powershell -File scripts\run.ps1 -Stage probe  -Profile qwen -Tag qwen4b_v2
#
# Read docs/START-HERE.md before the first run.

#Requires -Version 5.1
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('doctor', 'inputs', 'prompt', 'pilot', 'main', 'label', 'verify',
                 'baseline', 'probe', 'probe2', 'compare', 'status', 'all')]
    [string]$Stage,

    [string]$Profile = '',
    [string]$Tag = '',
    [string]$Model = '',
    [string]$InputFile = '',
    [string]$QuestionsFile = '',
    [string]$Ref = '',
    [ValidateSet('test', 'val')]
    [string]$Split = 'test',
    [switch]$ConfirmDeferred,
    [switch]$AllowGateWarnings,
    [switch]$DryRun,
    [ValidateSet('both', 'pilot', 'main')]
    [string]$PromptKind = 'both'
)

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'

$Root = Split-Path -Parent $PSScriptRoot
$Pipeline = Join-Path $Root 'pipeline'
$OutDir = Join-Path $Pipeline 'out'
$DataDir = Join-Path $Pipeline 'data'
$ProfilesFile = Join-Path $Root 'profiles.json'
$StateFile = Join-Path $Root 'PROJECT_STATE.json'

function Write-Step([string]$message) {
    Write-Host ""
    Write-Host "== $message" -ForegroundColor Cyan
}

function Fail([string]$message) {
    Write-Host ""
    Write-Host "ERROR: $message" -ForegroundColor Red
    exit 1
}

function Warn([string]$message) {
    Write-Host "WARNING: $message" -ForegroundColor Yellow
}

function Get-Json([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) { Fail "required file not found: $path" }
    return (Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json)
}

function Get-Prop($object, [string]$name) {
    if ($null -eq $object) { return $null }
    $p = $object.PSObject.Properties[$name]
    if ($null -eq $p) { return $null }
    return $p.Value
}

function Get-PythonPath {
    $venv = Join-Path $Pipeline '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venv) { return $venv }
    Warn "pipeline\.venv not found, falling back to 'python' on PATH"
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $cmd) { Fail "python not found: create the venv first (docs/SETUP.md)" }
    return $cmd.Source
}

function Resolve-RepoPath([string]$path) {
    if ([System.IO.Path]::IsPathRooted($path)) { return $path }
    return (Join-Path $Pipeline $path)
}

function Assert-Absent([string]$path, [string]$what) {
    if (Test-Path -LiteralPath $path) {
        Fail "$what already exists: $path`nThis runner never overwrites artifacts. Choose a new -Tag, or inspect the existing directory."
    }
}

function Assert-Tag {
    if (-not $Tag) { Fail '-Tag is required for this stage (a short name, e.g. qwen4b_v2)' }
    if ($Tag -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$') {
        Fail "invalid -Tag '$Tag': use letters, digits, underscore, dot or dash only"
    }
}

function Get-Profiles {
    return (Get-Json $ProfilesFile)
}

function Get-Profile([string]$id) {
    if (-not $id) {
        Fail "-Profile is required. Choose one of: $((Get-Profiles).profiles.PSObject.Properties.Name -join ', ')"
    }
    $profiles = Get-Profiles
    $p = $profiles.profiles.PSObject.Properties[$id]
    if ($null -eq $p) {
        Fail "unknown profile '$id'. Valid profiles: $($profiles.profiles.PSObject.Properties.Name -join ', ')"
    }
    return $p.Value
}

function Get-Flags($profileConfig, $profilesDoc) {
    $flags = [ordered]@{
        generate       = $true
        max_new_tokens = 64
        max_len        = 1024
        layers         = 'all'
        ngl            = 99
        no_template    = $false
        no_thinking    = $false
        thinking       = $false
        batch_rows     = 1
    }
    foreach ($name in $profilesDoc.defaults.flags.PSObject.Properties.Name) { $flags[$name] = $profilesDoc.defaults.flags.$name }
    foreach ($name in $profileConfig.flags.PSObject.Properties.Name) { $flags[$name] = $profileConfig.flags.$name }
    return $flags
}

function Assert-BinaryVersion([string]$exe, $entry) {
    if ($DryRun) { return }
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    $psi.Arguments = '--version'
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    if (-not $proc.Start()) { Fail "could not run --version: $exe" }
    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()
    $text = $stdout + "`n" + $stderr
    if ($proc.ExitCode -ne 0) { Fail "$exe --version exited with code $($proc.ExitCode): $text" }
    $build = $null
    $commit = $null
    if ($text -match 'build\s+(\d+),\s+commit\s+([0-9a-f]+)') {
        $build = [int]$Matches[1]
        $commit = $Matches[2]
    } elseif ($text -match 'version:\s*(\d+)\s+\(([0-9a-f]+)\)') {
        $build = [int]$Matches[1]
        $commit = $Matches[2]
    } else {
        Fail "could not parse --version output from $exe : $text"
    }
    if ($build -ne [int]$entry.build -or $commit -ne [string]$entry.commit) {
        Fail "binary version mismatch: got $commit / build $build, expected $($entry.commit) / build $($entry.build). Rebuild before running."
    }
}

function Get-Binary([string]$kind) {
    $state = Get-Json $StateFile
    $entry = if ($kind -eq 'legacy') { $state.binaries.legacy } else { $state.binaries.active }
    $exe = Join-Path $Root $entry.exe
    $dump = Join-Path $Root $entry.prompt_dump
    if ($kind -eq 'kaggle') { Fail "profile '$Profile' runs on Kaggle, not locally. See kaggle\README.md." }
    if (-not (Test-Path -LiteralPath $exe)) {
        Fail "binary not found: $exe`nRun scripts\bootstrap.ps1 -Build, or build it as described in docs/SETUP.md."
    }
    Assert-BinaryVersion $exe $entry
    return @{ exe = $exe; dump = $dump; commit = $entry.commit; build = $entry.build }
}

function Invoke-Python([string[]]$pyArgs) {
    $py = Get-PythonPath
    Write-Host "  python $($pyArgs -join ' ')" -ForegroundColor DarkGray
    if ($DryRun) {
        Write-Host "  (dry run: python not executed)" -ForegroundColor Yellow
        return
    }
    & $py @pyArgs
    if ($LASTEXITCODE -ne 0) { Fail "python step failed with exit code $LASTEXITCODE" }
}

function Quote-Arg([string]$value) {
    if ($value -match '[\s"]') { return '"' + ($value -replace '"', '\"') + '"' }
    return $value
}

function Invoke-Process([string]$exe, [string[]]$argv, [string]$logPath) {
    $argLine = ($argv | ForEach-Object { Quote-Arg $_ }) -join ' '
    $stdoutPath = "$logPath.stdout"
    try {
        $proc = Start-Process -FilePath $exe -ArgumentList $argLine -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $logPath
    }
    catch {
        Fail "could not start $exe : $($_.Exception.Message)"
    }
    if (Test-Path -LiteralPath $stdoutPath) {
        $extra = Get-Content -LiteralPath $stdoutPath -Encoding UTF8 -ErrorAction SilentlyContinue
        if ($null -ne $extra) { $extra | Add-Content -LiteralPath $logPath -Encoding UTF8 }
        Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
    }
    return $proc.ExitCode
}

function Invoke-Binary([string]$exe, [string[]]$argv, [string]$logPath) {
    $argLine = ($argv | ForEach-Object { Quote-Arg $_ }) -join ' '
    Write-Host "  $exe $argLine" -ForegroundColor DarkGray
    if ($DryRun) {
        Write-Host "  (dry run: binary not executed)" -ForegroundColor Yellow
        return
    }
    Assert-Absent $logPath 'log file'
    $code = Invoke-Process $exe $argv $logPath
    if ($code -ne 0) {
        Fail "binary exited with code $code. See $logPath"
    }
    Write-Host "  log: $logPath" -ForegroundColor DarkGray
}

function Build-BinaryArgs($flags, [string]$modelPath, [string]$inputPath, [string]$outPath) {
    $a = @('-m', $modelPath, '--batch', $inputPath, '--out', $outPath)
    if ($flags.generate) { $a += '--generate' }
    $a += @('--max-new-tokens', [string]$flags.max_new_tokens)
    $a += @('--max-len', [string]$flags.max_len)
    $a += @('--layers', [string]$flags.layers)
    $a += @('-ngl', [string]$flags.ngl)
    if ($flags.no_template) { $a += '--no-template' }
    if ($flags.no_thinking) { $a += '--no-thinking' }
    if ($flags.thinking) { $a += '--thinking' }
    if ([int]$flags.batch_rows -gt 1) { $a += @('--batch-rows', [string]$flags.batch_rows) }
    return $a
}

function Assert-NotBlocked([string]$id) {
    $state = Get-Json $StateFile
    foreach ($block in $state.authorization.blocked_actions) {
        if ($block.id -eq $id) {
            if ($ConfirmDeferred) {
                Warn "running a deferred action because -ConfirmDeferred was passed: $id"
                return
            }
            Fail "action '$id' is blocked: $($block.reason)`nIf the user explicitly asked for it, repeat the command with -ConfirmDeferred."
        }
    }
}

function Get-InputEntry($profileConfig, [string]$kind) {
    $entry = $profileConfig.inputs.PSObject.Properties[$kind]
    if ($null -eq $entry) { Fail "profile '$Profile' has no inputs.$kind entry" }
    return $entry.Value
}

function Resolve-InputPath($profileConfig, [string]$kind, [switch]$Build) {
    if ($InputFile) { return (Resolve-RepoPath $InputFile) }
    $entry = Get-InputEntry $profileConfig $kind
    $source = Resolve-RepoPath $entry.source
    $builder = Get-Prop $entry 'builder'
    if (-not $builder) { return $source }

    $built = Resolve-RepoPath (Get-Prop $entry 'built')
    if (Test-Path -LiteralPath $built) { return $built }
    if (-not $Build) {
        Fail "built input is missing: $built`nRun -Stage prompt first (it builds the file from $source)."
    }
    Assert-Absent $built 'built input'
    Write-Step "building input: $source -> $built"
    Invoke-Python @((Join-Path $Pipeline $builder), '--in', $source, '--out', $built)
    if ($DryRun) { return $built }
    if (-not (Test-Path -LiteralPath $built)) { Fail "builder did not create $built" }
    return $built
}

function Resolve-QuestionSource($profileConfig, [string]$kind) {
    if ($QuestionsFile) { return (Resolve-RepoPath $QuestionsFile) }
    if ($InputFile) {
        Fail "-InputFile requires -QuestionsFile for reproducible filtered labels (the generation input may be a prepared prompt)."
    }
    if ($null -eq $profileConfig) { return (Join-Path $DataDir 'main.jsonl') }
    $entry = Get-InputEntry $profileConfig $kind
    return (Resolve-RepoPath $entry.source)
}

function Assert-InputMatchesProfile($profileConfig, [string]$inputPath) {
    if ($DryRun) { return }
    $firstLine = Get-Content -LiteralPath $inputPath -TotalCount 1 -Encoding UTF8
    try { $firstRow = $firstLine | ConvertFrom-Json }
    catch { Fail "could not parse the first input row: $inputPath" }
    $text = [string](Get-Prop $firstRow 'question')
    $gptoss = $text -match '<\|start\|>system'
    $lfm = $text -match 'Final answer:\s*$'
    $id = $Profile
    if ($gptoss -and $id -ne 'gptoss20b') {
        Fail "this input is a pre-filled gpt-oss harmony prompt, but -Profile is '$id'. Use -Profile gptoss20b, and the runner will pass --no-template."
    }
    if ($lfm -and $id -notin @('lfm25-2.6b', 'lfm25-1.2b-thinking')) {
        Fail "this input ends with a 'Final answer:' cue, but -Profile is '$id'. Use an LFM final-prefill profile."
    }
    if ($id -eq 'gptoss20b' -and -not $gptoss) {
        Fail "profile gptoss20b needs the harmony final prompt built by make_gptoss_prompt.py; $inputPath does not look like one."
    }
}

function Invoke-Preflight([string]$inputPath, [string]$kind) {
    $expect = if ($kind -eq 'pilot') { 200 } else { $null }
    $pyArgs = @((Join-Path $Pipeline 'preflight.py'), '--input', $inputPath, '--require-instruction')
    if ($expect) { $pyArgs += @('--expect-rows', [string]$expect) }
    if ($kind -eq 'main') {
        $report = Join-Path $OutDir "prompt_$Tag.preflight.json"
        if (Test-Path -LiteralPath $report) {
            Warn "preflight report already exists, not overwriting: $report"
        } else {
            $pyArgs += @('--json', $report)
        }
    }
    Write-Step "preflight $kind input: $inputPath"
    Invoke-Python $pyArgs
}

function Invoke-PromptCheck([string]$kind, $profileConfig, $binary, [string]$inputPath) {
    Assert-Tag
    $dumpPath = Join-Path $OutDir "prompt_$Tag.$kind.txt"
    $recordPath = Join-Path $OutDir "prompt_$Tag.$kind.json"
    Assert-Absent $dumpPath "prompt dump"
    Assert-Absent $recordPath "prompt check record"
    if (-not $Model) { Fail '-Model is required for the prompt stage' }
    if (-not (Test-Path -LiteralPath $Model)) { Fail "model not found: $Model" }
    if (-not (Test-Path -LiteralPath $binary.dump)) {
        Fail "hs-prompt-dump not found: $($binary.dump)`nBuild the target hs-prompt-dump (docs/SETUP.md)."
    }
    $dumpExe = $binary.dump

    Write-Step "rendering the prompt for $kind (vocab only, no GPU work)"
    if ($DryRun) {
        Write-Host "  $dumpExe $Model > $dumpPath" -ForegroundColor DarkGray
        Write-Host "  (dry run: not executed)" -ForegroundColor Yellow
        return
    }
    $code = Invoke-Process $dumpExe @($Model) $dumpPath
    if ($code -ne 0) { Fail "hs-prompt-dump failed; see $dumpPath" }

    $dump = Get-Content -LiteralPath $dumpPath -Raw -Encoding UTF8
    $addBos = $null
    $bosId = $null
    $firstTok = $null
    if ($dump -match 'add_bos=(\d+)\s+bos=(-?\d+)') {
        $addBos = [int]$Matches[1]
        $bosId = [int]$Matches[2]
    }
    if ($dump -match 'tokens add_special=true\s+\((\d+)\):\s+(-?\d+)') { $firstTok = [int]$Matches[2] }
    if ($null -eq $addBos) { Fail "could not parse add_bos from $dumpPath" }

    Write-Host "  add_bos=$addBos bos=$bosId first_token=$firstTok" -ForegroundColor DarkGray
    if ($addBos -eq 1 -and ($null -eq $firstTok -or $firstTok -ne $bosId)) {
        Fail "the model requires BOS id $bosId, but the tokenized prompt starts with $firstTok.`nThis is the lost-<bos> failure mode. Do not run."
    }
    $expect = Get-Prop $profileConfig 'prompt_expectation'
    $wantBos = Get-Prop $expect 'add_bos'
    $wantFirst = Get-Prop $expect 'first_token'
    if ($null -ne $wantBos -and [int]$wantBos -ne $addBos) {
        Fail "profile '$Profile' expects add_bos=$wantBos, the model reports $addBos.`nDo not run: the prompt regime would differ from the reference run. Read docs/START-HERE.md, section 'BOS'."
    }
    if ($null -ne $wantFirst -and $null -ne $firstTok -and [int]$wantFirst -ne $firstTok) {
        Fail "profile '$Profile' expects the first prompt token to be $wantFirst, got $firstTok.`nThis is the Gemma 4 12B failure mode: the leading <bos> was lost. Stop and inspect $dumpPath."
    }
    Write-Host "  prompt check passed" -ForegroundColor Green

    if (-not (Test-Path -LiteralPath $inputPath)) { Fail "input not found: $inputPath" }
    $entry = Get-InputEntry $profileConfig $kind
    $builder = Get-Prop $entry 'builder'
    $built = Get-Prop $entry 'built'
    if ($builder -and $built) {
        $builtPath = Resolve-RepoPath $built
        $rebuild = Join-Path $OutDir "prompt_$Tag.$kind.rebuild.jsonl"
        Assert-Absent $rebuild "idempotency rebuild"
        Write-Step "builder idempotency check ($builder)"
        Invoke-Python @((Join-Path $Pipeline $builder), '--in', (Resolve-RepoPath $entry.source), '--out', $rebuild)
        $h1 = (Get-FileHash -LiteralPath $builtPath -Algorithm SHA256).Hash
        $h2 = (Get-FileHash -LiteralPath $rebuild -Algorithm SHA256).Hash
        if ($h1 -ne $h2) {
            Fail "the builder is not idempotent: $builtPath and $rebuild differ (`n  $h1`n  $h2)`nThe kept file is the one the reference run used; do not overwrite it. Inspect the builder."
        }
        Remove-Item -LiteralPath $rebuild
        Write-Host "  rebuild is byte-identical" -ForegroundColor Green
    }

    $modelInfo = Get-Item -LiteralPath $Model
    $record = [ordered]@{
        passed               = $true
        tag                  = $Tag
        kind                 = $kind
        profile              = $Profile
        model                = $Model
        model_length         = $modelInfo.Length
        model_last_write_utc = $modelInfo.LastWriteTimeUtc.ToString('o')
        input                = $inputPath
        input_sha256         = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash
        binary               = $binary.exe
        commit               = $binary.commit
        build                = $binary.build
        add_bos              = $addBos
        bos_id               = $bosId
        first_token          = $firstTok
        checked              = (Get-Date).ToUniversalTime().ToString('o')
    }
    $tmp = "$recordPath.tmp"
    ($record | ConvertTo-Json -Depth 5) | Out-File -FilePath $tmp -Encoding utf8
    Move-Item -LiteralPath $tmp -Destination $recordPath
    Write-Host "  prompt record: $recordPath" -ForegroundColor DarkGray
}

function Assert-PromptChecked([string]$kind, [string]$inputPath, $binary) {
    $recordPath = Join-Path $OutDir "prompt_$Tag.$kind.json"
    if ($DryRun -and -not (Test-Path -LiteralPath $recordPath)) {
        Write-Host "  (dry run: a real $kind run would require passed $recordPath)" -ForegroundColor Yellow
        return
    }
    if (-not (Test-Path -LiteralPath $recordPath)) {
        Fail "prompt check is missing: $recordPath`nRun -Stage prompt with this tag/profile/model before GPU work."
    }
    $record = Get-Json $recordPath
    if (-not $record.passed) { Fail "prompt check did not pass: $recordPath" }
    if ($record.profile -ne $Profile -or $record.model -ne $Model -or $record.input -ne $inputPath) {
        Fail "prompt check provenance differs from this run. Use a new tag and rerun -Stage prompt.`n  record profile/model/input: $($record.profile) / $($record.model) / $($record.input)`n  run    profile/model/input: $Profile / $Model / $inputPath"
    }
    if ($record.commit -ne $binary.commit -or [int]$record.build -ne [int]$binary.build) {
        Fail "prompt check used a different binary: $($record.commit)/$($record.build) vs $($binary.commit)/$($binary.build)"
    }
    $inputHash = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash
    if ($record.input_sha256 -ne $inputHash) {
        Fail "input changed after the prompt check: $inputPath`nUse a new tag and rerun -Stage prompt."
    }
    $modelInfo = Get-Item -LiteralPath $Model
    if ([int64]$record.model_length -ne [int64]$modelInfo.Length -or
        $record.model_last_write_utc -ne $modelInfo.LastWriteTimeUtc.ToString('o')) {
        Fail "model file changed after the prompt check: $Model`nUse a new tag and rerun -Stage prompt."
    }
    Write-Host "  prompt provenance verified: $recordPath" -ForegroundColor DarkGray
}

function Invoke-PilotGate([string]$runDir, [string]$inputPath, [string]$logPath, [string]$kind) {
    $gatePath = Join-Path $runDir 'gate.json'
    $pyArgs = @((Join-Path $Pipeline 'gate_report.py'),
        '--dir', $runDir,
        '--input', $inputPath,
        '--log', $logPath,
        '--profile', $Profile,
        '--stage', $kind,
        '--json', $gatePath)
    Write-Step "gate $kind"
    $py = Get-PythonPath
    $argLine = ($pyArgs | ForEach-Object { Quote-Arg $_ }) -join ' '
    $proc = Start-Process -FilePath $py -ArgumentList $argLine -NoNewWindow -Wait -PassThru
    return $proc.ExitCode
}

function Invoke-MetaPostprocessor($profileConfig, [string]$runDir) {
    $postprocessor = Get-Prop $profileConfig 'meta_postprocessor'
    if (-not $postprocessor) { return }
    if ($postprocessor -ne 'lfm_reasoning_final') {
        Fail "unknown meta_postprocessor '$postprocessor' in profile '$Profile'"
    }
    $meta = Join-Path $runDir 'meta.jsonl'
    $raw = Join-Path $runDir 'meta.raw.jsonl'
    Assert-Absent $raw 'raw meta backup'
    Move-Item -LiteralPath $meta -Destination $raw
    Write-Step "finalizing reasoning replay metadata"
    Invoke-Python @((Join-Path $Pipeline 'lfm_reasoning.py'), 'finalize-replay',
        '--replay-meta', $raw, '--out', $meta)
}

function Write-RunnerRecord([string]$runDir, $flags, [string]$inputPath,
                            [string]$questionsPath, $binary, [string]$kind) {
    if ($DryRun) { return }
    $record = [ordered]@{
        stage       = $kind
        tag         = $Tag
        profile     = $Profile
        model       = $Model
        input       = $inputPath
        input_sha256 = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash
        questions   = $questionsPath
        questions_sha256 = (Get-FileHash -LiteralPath $questionsPath -Algorithm SHA256).Hash
        binary      = $binary.exe
        commit      = $binary.commit
        build       = $binary.build
        flags       = $flags
        started     = (Get-Date).ToString('s')
        host        = $env:COMPUTERNAME
    }
    $path = Join-Path $runDir 'runner.json'
    ($record | ConvertTo-Json -Depth 5) | Out-File -FilePath $path -Encoding utf8
}

function Invoke-Generation([string]$kind) {
    Assert-Tag
    if ($InputFile -and -not $QuestionsFile) {
        Fail "-InputFile requires -QuestionsFile (original, unrendered questions used by label.py)."
    }
    $profilesDoc = Get-Profiles
    $profileConfig = Get-Profile $Profile
    $flags = Get-Flags $profileConfig $profilesDoc
    Assert-NotBlocked "run:$Profile"
    $blocked = Get-Prop $profileConfig 'blocked'
    if ($null -ne $blocked) {
        $blockedId = Get-Prop $blocked 'id'
        if (-not $ConfirmDeferred) {
            Fail "profile '$Profile' is blocked: $(Get-Prop $blocked 'reason')`nIf the user explicitly asked for it, repeat with -ConfirmDeferred."
        }
        Warn "profile is normally blocked; continuing because -ConfirmDeferred was passed ($blockedId)"
    }
    if (-not $Model) { Fail "-Model is required for stage $kind (path to the .gguf)" }
    if (-not (Test-Path -LiteralPath $Model)) { Fail "model not found: $Model" }

    $binary = Get-Binary (Get-Prop $profileConfig 'binary')
    $inputPath = Resolve-InputPath $profileConfig $kind -Build
    $questionSource = Resolve-QuestionSource $profileConfig $kind
    if (-not (Test-Path -LiteralPath $questionSource)) {
        Fail "original question source is missing: $questionSource"
    }
    Assert-InputMatchesProfile $profileConfig $inputPath
    Assert-PromptChecked $kind $inputPath $binary
    Invoke-Preflight $inputPath $kind

    $runName = "${kind}_$Tag"
    $runDir = Join-Path $OutDir $runName
    $logPath = Join-Path $OutDir "$runName.log"
    Assert-Absent $runDir 'run directory'
    Assert-Absent $logPath 'log file'

    if ($kind -eq 'main') {
        $pilotDir = Join-Path $OutDir "pilot_$Tag"
        $pilotGate = Join-Path $pilotDir 'gate.json'
        if ($DryRun) {
            Write-Host "  (dry run: a real main run would require passed $pilotGate)" -ForegroundColor Yellow
        } else {
        if (-not (Test-Path -LiteralPath $pilotGate)) {
            Fail "no pilot gate: $pilotGate`nStage main requires a passed pilot for the same -Tag. Run -Stage pilot first."
        }
        $gate = Get-Json $pilotGate
        if (-not $gate.passed) { Fail "pilot gate failed for tag '$Tag'; main is not allowed. See $pilotGate" }
        $pilotLabels = Join-Path $pilotDir 'labels.csv'
        if (-not (Test-Path -LiteralPath $pilotLabels)) {
            Fail "pilot labels are missing: $pilotLabels`nThe pilot did not reach its final class-balance gate. Use a new tag."
        }
        $classCheck = @($gate.checks | Where-Object { $_.check -eq 'both label classes present' })
        if ($classCheck.Count -ne 1 -or -not $classCheck[0].ok) {
            Fail "pilot gate has no passed class-balance check: $pilotGate`nRerun the pilot with a new tag."
        }
        $pilotRunner = Join-Path $pilotDir 'runner.json'
        if (-not (Test-Path -LiteralPath $pilotRunner)) {
            Fail "pilot provenance is missing: $pilotRunner`nRun the pilot through scripts\run.ps1 with a new tag; model/profile/flags cannot otherwise be verified."
        }
        if (Test-Path -LiteralPath $pilotRunner) {
            $prev = Get-Json $pilotRunner
            if ($prev.model -ne $Model) {
                Fail "the pilot used a different model:`n  pilot: $($prev.model)`n  main:  $Model`nRun a new pilot for this model."
            }
            if ($prev.profile -ne $Profile) {
                Fail "the pilot used a different profile:`n  pilot: $($prev.profile)`n  main:  $Profile`nA profile changes prompt handling and flags. Run a new pilot with the main profile."
            }
            if ($prev.commit -ne $binary.commit -or [int]$prev.build -ne [int]$binary.build) {
                Fail "the pilot used a different binary:`n  pilot: $($prev.commit) / build $($prev.build)`n  main:  $($binary.commit) / build $($binary.build)`nRebuild if needed, then run a new pilot."
            }
            $pilotFlags = $prev.flags | ConvertTo-Json -Depth 5 -Compress
            $mainFlags = $flags | ConvertTo-Json -Depth 5 -Compress
            if ($pilotFlags -ne $mainFlags) {
                Fail "the pilot and main flags differ:`n  pilot: $pilotFlags`n  main:  $mainFlags`nRun a new pilot with exactly the main flags."
            }
        }
        Write-Host "  pilot gate: $($gate.summary.rows) rows, $($gate.summary.truncated) truncated, estimate $($gate.summary.estimated_hours_6000) h" -ForegroundColor DarkGray
        }
    }

    if (-not $DryRun) {
        New-Item -ItemType Directory -Path $runDir -Force | Out-Null
        Write-RunnerRecord $runDir $flags $inputPath $questionSource $binary $kind
    }

    $argv = Build-BinaryArgs $flags $Model $inputPath $runDir
    Write-Step "$kind run: $runName"
    Invoke-Binary $binary.exe $argv $logPath

    if ($DryRun) { return }

    Invoke-MetaPostprocessor $profileConfig $runDir

    $gateCode = Invoke-PilotGate $runDir $inputPath $logPath $kind
    if ($gateCode -ne 0) {
        if ($AllowGateWarnings) {
            Warn "-AllowGateWarnings cannot override hard gate failures; warnings already pass without this switch"
        }
        Fail "gate failed for $runName. Read $(Join-Path $runDir 'gate.json') before continuing.`nFor a pilot this means: do not start main."
    }

    if ($kind -eq 'pilot') {
        $pilotLabels = Join-Path $runDir 'labels.csv'
        Assert-QuestionsMatchRunner $runDir $questionSource
        Assert-Absent $pilotLabels 'pilot labels.csv'
        Write-Step "labeling pilot for the class-balance gate"
        Invoke-Python @((Join-Path $Pipeline 'label.py'),
            '--meta', (Join-Path $runDir 'meta.jsonl'), '--out', $pilotLabels,
            '--questions', $questionSource)
        $gateCode = Invoke-PilotGate $runDir $inputPath $logPath $kind
        if ($gateCode -ne 0) {
            Fail "pilot failed after labeling (missing class or unexplained null labels). See $(Join-Path $runDir 'gate.json'). Do not start main."
        }
    }

    Write-Host ""
    Write-Host "$kind finished: $runDir" -ForegroundColor Green
    if ($kind -eq 'pilot') {
        Write-Host "Next: read 10 answers from $runDir\meta.jsonl, then -Stage main with the same -Tag." -ForegroundColor Green
    } else {
        Write-Host "Next: -Stage label, then verify, baseline and probe." -ForegroundColor Green
    }
}

function Get-RunDir([string]$kind) {
    Assert-Tag
    $dir = Join-Path $OutDir "${kind}_$Tag"
    if (-not (Test-Path -LiteralPath $dir)) {
        Fail "run directory not found: $dir"
    }
    return $dir
}

function Assert-RunnerGatePassed([string]$dir) {
    # Historical/manual directories predate runner.json and remain inspectable. A
    # directory created by this runner, however, must never advance to analysis
    # after its own hard gate failed.
    $runner = Join-Path $dir 'runner.json'
    if (-not (Test-Path -LiteralPath $runner)) { return }
    $gatePath = Join-Path $dir 'gate.json'
    if (-not (Test-Path -LiteralPath $gatePath)) {
        Fail "runner-created directory has no gate.json: $dir"
    }
    $gate = Get-Json $gatePath
    if (-not $gate.passed) {
        Fail "run gate failed for $dir; baseline/probe are blocked. Keep the directory as evidence and use a new tag."
    }
}

function Assert-QuestionsMatchRunner([string]$dir, [string]$questionSource) {
    $runnerPath = Join-Path $dir 'runner.json'
    if (-not (Test-Path -LiteralPath $runnerPath)) { return }
    $runnerRecord = Get-Json $runnerPath
    $questionHash = (Get-FileHash -LiteralPath $questionSource -Algorithm SHA256).Hash
    if ($runnerRecord.questions -ne $questionSource -or $runnerRecord.questions_sha256 -ne $questionHash) {
        Fail "original questions changed since generation. Keep the run untouched and investigate:`n  runner: $($runnerRecord.questions) / $($runnerRecord.questions_sha256)`n  label:  $questionSource / $questionHash"
    }
}

function Resolve-VerifyInput($profileConfig, [string]$kind) {
    if ($InputFile) { return (Resolve-RepoPath $InputFile) }
    if ($null -eq $profileConfig) { return $null }
    $entry = Get-InputEntry $profileConfig $kind
    $built = Get-Prop $entry 'built'
    if ($built) {
        $p = Resolve-RepoPath $built
        if (Test-Path -LiteralPath $p) { return $p }
    }
    $p = Resolve-RepoPath $entry.source
    if (Test-Path -LiteralPath $p) { return $p }
    return $null
}

function Invoke-LabelStage {
    $dir = Get-RunDir 'main'
    $labels = Join-Path $dir 'labels.csv'
    $profileConfig = if ($Profile) { Get-Profile $Profile } else { $null }
    $questionSource = Resolve-QuestionSource $profileConfig 'main'
    if (-not (Test-Path -LiteralPath $questionSource)) {
        Fail "original question source is missing: $questionSource"
    }
    Assert-QuestionsMatchRunner $dir $questionSource
    Assert-Absent $labels 'labels.csv'
    Write-Step "labeling $dir"
    Invoke-Python @((Join-Path $Pipeline 'label.py'), '--meta', (Join-Path $dir 'meta.jsonl'),
        '--out', $labels, '--questions', $questionSource)
}

function Invoke-VerifyStage {
    $dir = Get-RunDir 'main'
    $labels = Join-Path $dir 'labels.csv'
    if (-not (Test-Path -LiteralPath $labels)) { Fail "labels.csv missing: run -Stage label first" }
    $verify = Join-Path $dir 'verify.json'
    Assert-Absent $verify 'verify.json'
    $profileConfig = if ($Profile) { Get-Profile $Profile } else { $null }
    $inputPath = Resolve-VerifyInput $profileConfig 'main'
    $pyArgs = @((Join-Path $Pipeline 'verify_run.py'), '--dir', $dir, '--json', $verify, '--full')
    if ($inputPath) { $pyArgs += @('--input', $inputPath) }
    Write-Step "verify $dir"
    Invoke-Python $pyArgs
}

function Invoke-BaselineStage {
    $dir = Get-RunDir 'main'
    Assert-RunnerGatePassed $dir
    if (-not (Test-Path -LiteralPath (Join-Path $dir 'labels.csv'))) { Fail "labels.csv missing: run -Stage label first" }
    $verify = Join-Path $dir 'verify.json'
    if (-not (Test-Path -LiteralPath $verify)) { Fail "verify.json missing: run -Stage verify before baseline" }
    $verifyReport = Get-Json $verify
    if ($verifyReport.fails -ne 0) { Fail "verify_run.py reported $($verifyReport.fails) failures; baseline is blocked" }
    $out = Join-Path $dir 'baseline.json'
    $scoresOut = Join-Path $dir 'baseline_scores.csv'
    Assert-Absent $out 'baseline.json'
    Assert-Absent $scoresOut 'baseline_scores.csv'
    Write-Step "baseline $dir"
    Invoke-Python @((Join-Path $Pipeline 'baseline.py'),
        '--meta', (Join-Path $dir 'meta.jsonl'),
        '--labels', (Join-Path $dir 'labels.csv'),
        '--out', $out,
        '--scores-out', $scoresOut)
}

function Invoke-ProbeStage([switch]$Repeat) {
    $dir = Get-RunDir 'main'
    Assert-RunnerGatePassed $dir
    if (-not (Test-Path -LiteralPath (Join-Path $dir 'baseline.json'))) {
        Fail "baseline.json missing: run -Stage baseline before probe"
    }
    $verify = Join-Path $dir 'verify.json'
    if (-not (Test-Path -LiteralPath $verify)) { Fail "verify.json missing: run -Stage verify first" }
    $v = Get-Json $verify
    if ($v.fails -ne 0) { Fail "verify_run.py reported $($v.fails) failures; fix or explain them before probe" }
    if ($v.warns -gt 0) { Warn "verify.json has $($v.warns) warnings (expected for pruned l_out models)" }

    Assert-NotBlocked "probe:main_$Tag"

    $prefix = if ($Repeat) { 'probe2' } else { 'probe' }
    $out = Join-Path $dir "$prefix.json"
    $curveOut = Join-Path $dir "${prefix}_curve.csv"
    $scoresOut = Join-Path $dir "${prefix}_scores.csv"
    $modelOut = Join-Path $dir "${prefix}_model.npz"
    if (-not $Repeat) { $modelOut = Join-Path $dir 'probe_model.npz' }
    if ($Repeat) {
        foreach ($required in @('probe.json', 'probe_curve.csv', 'probe_scores.csv', 'probe_model.npz')) {
            if (-not (Test-Path -LiteralPath (Join-Path $dir $required))) {
                Fail "$required is missing: run -Stage probe before probe2"
            }
        }
    }
    Assert-Absent $out "$prefix.json"
    Assert-Absent $curveOut "${prefix}_curve.csv"
    Assert-Absent $scoresOut "${prefix}_scores.csv"
    Assert-Absent $modelOut "${prefix}_model.npz"
    $pyArgs = @((Join-Path $Pipeline 'probe.py'),
        '--hidden-a', (Join-Path $dir 'hidden_a.npy'),
        '--hidden-c', (Join-Path $dir 'hidden_c.npy'),
        '--meta', (Join-Path $dir 'meta.jsonl'),
        '--labels', (Join-Path $dir 'labels.csv'),
        '--out', $out,
        '--curve-out', $curveOut,
        '--scores-out', $scoresOut,
        '--model-out', $modelOut)
    Write-Step "$prefix $dir"
    Invoke-Python $pyArgs

    if ($Repeat -and -not $DryRun) {
        $pairs = @(
            @((Join-Path $dir 'probe_curve.csv'), (Join-Path $dir 'probe2_curve.csv')),
            @((Join-Path $dir 'probe_scores.csv'), (Join-Path $dir 'probe2_scores.csv'))
        )
        foreach ($pair in $pairs) {
            $h1 = (Get-FileHash -LiteralPath $pair[0] -Algorithm SHA256).Hash
            $h2 = (Get-FileHash -LiteralPath $pair[1] -Algorithm SHA256).Hash
            if ($h1 -ne $h2) { Fail "probe repeat differs: $($pair[0]) vs $($pair[1])" }
            Write-Host "  identical: $(Split-Path -Leaf $pair[0])" -ForegroundColor Green
        }
    }
}

function Invoke-CompareStage {
    Assert-Tag
    if (-not $Ref) { Fail '-Ref is required: path to the reference probe_scores.csv' }
    $refPath = Resolve-RepoPath $Ref
    if (-not (Test-Path -LiteralPath $refPath)) { Fail "reference not found: $refPath" }
    $dir = Get-RunDir 'main'
    $mine = Join-Path $dir 'probe_scores.csv'
    if (-not (Test-Path -LiteralPath $mine)) { Fail "probe_scores.csv missing in $dir" }
    $refName = Split-Path -Leaf (Split-Path -Parent $refPath)
    $out = Join-Path $OutDir "compare_${refName}_${Tag}_$Split.json"
    Assert-Absent $out 'comparison output'
    Write-Step "compare $refName vs $Tag ($Split)"
    Invoke-Python @((Join-Path $Pipeline 'compare_probes.py'),
        '--a', $refPath, '--b', $mine, '--split', $Split, '--out', $out)
}

function Invoke-DoctorStage {
    $pyArgs = @((Join-Path $Root 'scripts\doctor.py'))
    if ($Model) { $pyArgs += @('--model', $Model) }
    if ($Profile) { $pyArgs += @('--profile', $Profile) }
    if ($InputFile) { $pyArgs += @('--input', (Resolve-RepoPath $InputFile)) }
    Write-Step "doctor"
    Invoke-Python $pyArgs
}

function Invoke-StatusStage {
    Write-Step "repository state"
    Invoke-Python @((Join-Path $Root 'scripts\build_runs.py'), '--check')
    Invoke-Python @((Join-Path $Root 'scripts\check_state.py'))
}

function Invoke-InputsStage {
    if (-not $Profile) { Fail '-Profile is required for stage inputs' }
    Assert-Tag
    $profileConfig = Get-Profile $Profile
    foreach ($kind in @('pilot', 'main')) {
        $path = Resolve-InputPath $profileConfig $kind -Build
        Invoke-Preflight $path $kind
    }
}

$started = Get-Date

switch ($Stage) {
    'doctor'   { Invoke-DoctorStage }
    'status'   { Invoke-StatusStage }
    'inputs'   { Invoke-InputsStage }
    'prompt'   {
        Assert-Tag
        $profileConfig = Get-Profile $Profile
        $binary = Get-Binary (Get-Prop $profileConfig 'binary')
        $promptKinds = if ($PromptKind -eq 'both') { @('pilot', 'main') } else { @($PromptKind) }
        foreach ($kind in $promptKinds) {
            $path = Resolve-InputPath $profileConfig $kind -Build
            Assert-InputMatchesProfile $profileConfig $path
            Invoke-PromptCheck $kind $profileConfig $binary $path
        }
    }
    'pilot'    { Invoke-Generation 'pilot' }
    'main'     { Invoke-Generation 'main' }
    'label'    { Invoke-LabelStage }
    'verify'   { Invoke-VerifyStage }
    'baseline' { Invoke-BaselineStage }
    'probe'    { Invoke-ProbeStage }
    'probe2'   { Invoke-ProbeStage -Repeat }
    'compare'  { Invoke-CompareStage }
    'all'      {
        Invoke-DoctorStage
        $profileConfig = Get-Profile $Profile
        $binary = Get-Binary (Get-Prop $profileConfig 'binary')
        foreach ($kind in @('pilot', 'main')) {
            $path = Resolve-InputPath $profileConfig $kind -Build
            Assert-InputMatchesProfile $profileConfig $path
            Invoke-PromptCheck $kind $profileConfig $binary $path
        }
        Invoke-Generation 'pilot'
        Invoke-Generation 'main'
        if ($DryRun) {
            Write-Step 'downstream stages'
            Write-Host '  (dry run: label -> verify -> baseline -> probe require files produced by pilot/main)' -ForegroundColor Yellow
        } else {
            Invoke-LabelStage
            Invoke-VerifyStage
            Invoke-BaselineStage
            Invoke-ProbeStage
        }
    }
}

$elapsed = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
Write-Host ""
Write-Host "stage '$Stage' finished in ${elapsed}s" -ForegroundColor Green
