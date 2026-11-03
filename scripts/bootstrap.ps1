# Bootstrap a clean clone of this repository so that it can run.
#
# A fresh clone contains only this project's own files: the two llama.cpp copies
# hold nothing except examples/hidden-states/* and one line in examples/CMakeLists.txt
# (the rest is ignored by .gitignore). This script downloads the two pinned upstream
# source archives and lays them under our files without overwriting them.
#
# Examples:
#   powershell -File scripts\bootstrap.ps1                  # sources + python venv
#   powershell -File scripts\bootstrap.ps1 -DryRun          # print the plan only
#   powershell -File scripts\bootstrap.ps1 -Build           # also build CUDA binaries
#   powershell -File scripts\bootstrap.ps1 -Build -NoCuda   # CPU-only build
#   powershell -File scripts\bootstrap.ps1 -Only legacy     # only the old pin
#
# It never deletes build\ or pipeline\out: an existing tree is left alone.

#Requires -Version 5.1
param(
    [ValidateSet('both', 'active', 'legacy')]
    [string]$Only = 'both',
    [switch]$Build,
    [switch]$NoCuda,
    [switch]$NoVenv,
    [switch]$SkipFetch,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$Tarball = 'https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{0}'

# Same five files in both copies: three sources plus the one-line edit in
# examples/CMakeLists.txt that registers the example. The archive must not
# overwrite any of them.
$TrackedFiles = @(
    'examples/CMakeLists.txt',
    'examples/hidden-states/CMakeLists.txt',
    'examples/hidden-states/hidden-states.cpp',
    'examples/hidden-states/load_example.py',
    'examples/hidden-states/prompt_dump.cpp'
)
$SkipFiles = @('examples\CMakeLists.txt')
$SkipDirs = @('examples\hidden-states')

$Copies = @{
    active = @{ dir = 'llama.cpp-hiddenstates-new'; commit = '785322d17'; build = 12033 }
    legacy = @{ dir = 'llama.cpp-hiddenstates'; commit = 'd4cff114c'; build = 9869 }
}

function Say([string]$message) { Write-Host $message }
function Step([string]$message) { Write-Host ""; Write-Host "== $message" -ForegroundColor Cyan }
function Fail([string]$message) { Write-Host ""; Write-Host "ERROR: $message" -ForegroundColor Red; exit 1 }
function Warn([string]$message) { Write-Host "WARNING: $message" -ForegroundColor Yellow }

function Require-Command([string]$name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Fail "$name not found on PATH. Install it first ($name --version)."
    }
}

function Invoke-Checked([string]$exe, [string[]]$argv, [string]$workdir = $Root) {
    Say "  $exe $($argv -join ' ')"
    if ($DryRun) { return }
    Push-Location -LiteralPath $workdir
    try {
        & $exe @argv
        if ($LASTEXITCODE -ne 0) { Fail "$exe exited with code $LASTEXITCODE" }
    }
    finally { Pop-Location }
}

function Test-UpstreamPresent([string]$dir) {
    return (Test-Path -LiteralPath (Join-Path $dir 'src\llama-context.cpp')) -and
           (Test-Path -LiteralPath (Join-Path $dir 'common\chat.cpp'))
}

function Assert-OurFiles([string]$dir) {
    $missing = @()
    foreach ($rel in $TrackedFiles) {
        $p = Join-Path $dir $rel
        if (-not (Test-Path -LiteralPath $p)) { $missing += $rel }
    }
    if ($missing.Count -gt 0) {
        Fail "missing our files in ${dir}:`n  $($missing -join "`n  ")`nThey are tracked in the repository. Restore them from a git clone with:`n  git checkout -- `"<path>`""
    }
    $cmake = Join-Path $dir 'examples\CMakeLists.txt'
    $text = Get-Content -LiteralPath $cmake -Raw -Encoding UTF8
    $count = ([regex]::Matches($text, 'add_subdirectory\(hidden-states\)')).Count
    if ($count -ne 1) {
        Fail "$cmake has $count 'add_subdirectory(hidden-states)' lines, expected exactly 1.`nRestore it from the repository; do not hand-edit."
    }
}

function Copy-ArchiveTree([string]$from, [string]$to) {
    # Copy everything except our five files: they are already in the target and
    # must keep their repository content (examples/CMakeLists.txt differs from
    # upstream by the registration line).
    $robocopy = Get-Command robocopy -ErrorAction SilentlyContinue
    if ($robocopy) {
        $argv = @($from, $to, '/E', '/NFL', '/NDL', '/NJH', '/NJS', '/NP', '/R:2', '/W:1')
        foreach ($f in $SkipFiles) { $argv += @('/XF', (Join-Path $from $f)) }
        foreach ($d in $SkipDirs) { $argv += @('/XD', (Join-Path $from $d)) }
        Say "  robocopy $from -> $to (our files excluded)"
        if ($DryRun) { return }
        & robocopy @argv | Out-Null
        # robocopy: 0-7 are success, 8+ are errors.
        if ($LASTEXITCODE -ge 8) { Fail "robocopy exited with code $LASTEXITCODE" }
        $global:LASTEXITCODE = 0
        return
    }
    Say "  copying $from -> $to (robocopy not found, using a protected file walk)"
    if ($DryRun) { return }
    Get-ChildItem -LiteralPath $from -Recurse -File -Force | ForEach-Object {
        $rel = $_.FullName.Substring($from.Length).TrimStart('\', '/')
        $protected = $SkipFiles -contains $rel
        foreach ($skipDir in $SkipDirs) {
            if ($rel -eq $skipDir -or $rel.StartsWith($skipDir + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
                $protected = $true
            }
        }
        if ($protected) { return }
        $dest = Join-Path $to $rel
        $parent = Split-Path -Parent $dest
        if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
        Copy-Item -LiteralPath $_.FullName -Destination $dest -Force
    }
}

function Install-Archive([string]$dir, [string]$commit) {
    Step "installing llama.cpp $commit into $dir"
    Require-Command 'tar'
    $stamp = [guid]::NewGuid().ToString('N').Substring(0, 8)
    $work = Join-Path ([System.IO.Path]::GetTempPath()) "hs-bootstrap-$stamp"
    $tgz = Join-Path $work 'src.tgz'
    $staging = Join-Path $work 'x'
    Say "  download $($Tarball -f $commit)"
    if ($DryRun) { return }
    New-Item -ItemType Directory -Path $staging -Force | Out-Null
    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri ($Tarball -f $commit) -OutFile $tgz -UseBasicParsing -TimeoutSec 900
        Say "  archive: $([math]::Round((Get-Item $tgz).Length / 1MB, 1)) MB"
        & tar -xzf $tgz -C $staging
        if ($LASTEXITCODE -ne 0) { Fail "tar failed with code $LASTEXITCODE" }
        $inner = Get-ChildItem -LiteralPath $staging -Directory | Select-Object -First 1
        if ($null -eq $inner) { Fail "the archive did not contain a directory" }
        if (-not (Test-Path -LiteralPath (Join-Path $inner.FullName 'src\llama-context.cpp'))) {
            Fail "the archive does not look like llama.cpp: $($inner.FullName)"
        }
        if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        Copy-ArchiveTree $inner.FullName $dir
    }
    finally {
        if (Test-Path -LiteralPath $work) { Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue }
    }
    Say "  installed $commit"
}

function Test-Venv {
    Step 'python environment'
    $venv = Join-Path $Root 'pipeline\.venv'
    $py = Join-Path $venv 'Scripts\python.exe'
    if (Test-Path -LiteralPath $py) {
        Say "  venv already exists: $venv"
    } else {
        Require-Command 'python'
        Invoke-Checked 'python' @('-m', 'venv', $venv)
    }
    $req = Join-Path $Root 'pipeline\requirements.txt'
    if (-not (Test-Path -LiteralPath $req)) { Fail "missing $req" }
    Invoke-Checked $py @('-m', 'pip', 'install', '--upgrade', 'pip', '--quiet')
    Invoke-Checked $py @('-m', 'pip', 'install', '-r', $req)
}

function Build-Copy([string]$name, [string]$dir, [string]$commit, [int]$number) {
    Step "building $name ($commit, build $number)"
    Require-Command 'cmake'
    $build = Join-Path $dir 'build'
    $cuda = if ($NoCuda) { 'OFF' } else { 'ON' }
    $cfg = @(
        '-S', $dir, '-B', $build,
        "-DGGML_CUDA=$cuda",
        "-DLLAMA_BUILD_COMMIT=$commit",
        "-DLLAMA_BUILD_NUMBER=$number"
    )
    if (-not $NoCuda) {
        $cfg += '-DGGML_CUDA_GRAPHS=ON'
        $cfg += '-DGGML_CUDA_NCCL=ON'
    }
    Invoke-Checked 'cmake' $cfg
    Invoke-Checked 'cmake' @('--build', $build, '--config', 'Release', '--target', 'llama-hidden-states', '--target', 'hs-prompt-dump', '--', '/m')

    $exe = Join-Path $build 'bin\Release\llama-hidden-states.exe'
    if (-not $DryRun) {
        if (-not (Test-Path -LiteralPath $exe)) { Fail "build finished but $exe is missing" }
        Say "  built: $exe"
        & $exe --version
    }
}

Step 'bootstrap'
Say "repository: $Root"
if ($DryRun) { Say "(dry run: nothing is written)" }

$names = if ($Only -eq 'both') { @('active', 'legacy') } else { @($Only) }
foreach ($name in $names) {
    $copy = $Copies[$name]
    $dir = Join-Path $Root $copy.dir
    Step "checking $($copy.dir) ($($copy.commit))"
    if (Test-UpstreamPresent $dir) {
        Say "  upstream sources already present, nothing downloaded"
    } elseif ($SkipFetch) {
        Warn "no upstream sources and -SkipFetch was passed"
    } else {
        Install-Archive $dir $copy.commit
    }
    Assert-OurFiles $dir
    Say "  our five files are in place"
}

if (-not $NoVenv) { Test-Venv }

if ($Build) {
    foreach ($name in $names) {
        $copy = $Copies[$name]
        $dir = Join-Path $Root $copy.dir
        $exe = Join-Path $dir 'build\bin\Release\llama-hidden-states.exe'
        $dump = Join-Path $dir 'build\bin\Release\hs-prompt-dump.exe'
        if ((Test-Path -LiteralPath $exe) -and (Test-Path -LiteralPath $dump) -and -not $DryRun) {
            Step "build for $name already exists"
            Say "  $exe"
            Say "  $dump"
            Say "  delete $($copy.dir)\build and rerun with -Build to rebuild"
            continue
        }
        Build-Copy $name $copy.dir $copy.commit $copy.build
    }
}

Step 'next steps'
Say '  powershell -File scripts\run.ps1 -Stage doctor -Profile standard -Model <path.gguf>'
Say '  powershell -File scripts\run.ps1 -Stage status'
Say '  read docs\START-HERE.md, then AGENTS.md'
