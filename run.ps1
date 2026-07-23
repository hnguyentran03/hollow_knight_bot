# One-command setup + launch (Windows). Safe to run every time:
#   powershell -ExecutionPolicy Bypass -File run.ps1 [extra dashboard args]
# First run builds/installs the mod and creates the Python env; later runs
# skip whatever is already up to date and go straight to the dashboard.
# macOS users: run.sh is the same thing for bash.
param(
    [string]$HKManaged = "C:\Program Files (x86)\Steam\steamapps\common\Hollow Knight\hollow_knight_Data\Managed",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$DashboardArgs = @()
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$installedDll = Join-Path $HKManaged "Mods\HKRLBot\HKRLBot.dll"
$venvPy = "trainer\.venv\Scripts\python.exe"
$reqFile = "trainer\requirements.txt"
$reqStamp = "trainer\.venv\requirements-stamp"

function Find-Python {
    # Returns the interpreter invocation as an array (e.g. @("py", "-3")),
    # or $null when no Python 3.11+ is on PATH.
    # Stderr of a native command redirected under ErrorActionPreference=Stop
    # throws in Windows PowerShell 5.1, so relax it locally.
    $ErrorActionPreference = "Continue"
    foreach ($cmd in @("python", "py")) {
        if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { continue }
        $verArgs = if ($cmd -eq "py") { @("-3") } else { @() }
        & $cmd @verArgs -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { return (@($cmd) + $verArgs) }
    }
    return $null
}

function Test-HasModdingApi([string]$dll) {
    # The Modding API lives inside the patched Assembly-CSharp.dll; the type
    # name "ModHooks" appears in its metadata and never in the vanilla dll.
    if (-not (Test-Path $dll)) { return $false }
    return [bool](Select-String -Path $dll -Pattern "ModHooks" -Quiet)
}

# ---- figure out what this launch actually needs -----------------------------

$needVenv = -not (Test-Path $venvPy)
$needPip = $needVenv -or -not (Test-Path $reqStamp) -or
    ((Get-Content -Raw $reqFile) -ne (Get-Content -Raw $reqStamp))

# Rebuild when the installed dll is missing or any mod source is newer than it
# (generated sources under bin\ and obj\ don't count).
$needBuild = -not (Test-Path $installedDll)
if (-not $needBuild) {
    $dllTime = (Get-Item $installedDll).LastWriteTime
    $newer = Get-ChildItem mod -Recurse -Include *.cs, *.csproj |
        Where-Object { $_.FullName -notmatch '\\(bin|obj)\\' -and $_.LastWriteTime -gt $dllTime }
    if ($newer) { $needBuild = $true }
}

# ---- dependency check: collect every problem, then report them together -----

$missing = @()

if (-not (Test-Path $HKManaged)) {
    $missing += "Hollow Knight was not found at the default Steam location:
      $HKManaged
    Install it via Steam, or if it lives in a different Steam library,
    pass its Managed dir: powershell -File run.ps1 -HKManaged <path>"
}

$python = $null
if ($needVenv) {
    $python = Find-Python
    if (-not $python) {
        $missing += "Python 3.11+ is required but was not found (or is too old).
    Install it from https://www.python.org/downloads/ (check 'Add python.exe to PATH')."
    }
}

if ($needBuild) {
    if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
        $missing += ".NET SDK is required to build the mod but 'dotnet' was not found.
    Install it from https://dotnet.microsoft.com/download (or: winget install Microsoft.DotNet.SDK.8)"
    }
    if ((Test-Path $HKManaged) -and
        -not (Test-HasModdingApi (Join-Path $HKManaged "Assembly-CSharp.dll")) -and
        -not (Test-HasModdingApi (Join-Path $HKManaged "Assembly-CSharp.dll.m"))) {
        $missing += "The Hollow Knight Modding API is not installed (Assembly-CSharp.dll is vanilla).
    Install it with Lumafly (https://themulhima.github.io/Lumafly/) and run this script again."
    }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "Cannot start yet -- $($missing.Count) thing(s) need attention:" -ForegroundColor Red
    $n = 0
    foreach ($m in $missing) {
        $n++
        Write-Host ""
        Write-Host "  $n. $m" -ForegroundColor Red
    }
    Write-Host ""
    exit 1
}

# ---- setup steps (each skipped when already done) ---------------------------

if ($needVenv) {
    Write-Host "==> Creating Python environment (first run only)..."
    $venvArgs = @($python | Select-Object -Skip 1) + @("-m", "venv", "trainer\.venv")
    & $python[0] @venvArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($needPip) {
    Write-Host "==> Installing Python dependencies (takes a few minutes the first time)..."
    & $venvPy -m pip install -q -r $reqFile
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Copy-Item $reqFile $reqStamp -Force
}

if ($needBuild) {
    Write-Host "==> Building and installing the HKRLBot mod..."
    powershell -ExecutionPolicy Bypass -File mod\build.ps1 -HKManaged $HKManaged
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# ---- launch -----------------------------------------------------------------

Write-Host "==> Starting the dashboard (Ctrl-C stops it; training runs it launched keep going)..."
Set-Location trainer
& .\.venv\Scripts\python.exe scripts\dashboard.py --open @DashboardArgs
exit $LASTEXITCODE
