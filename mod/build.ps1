# Windows counterpart of build.sh: build the mod and install it into the
# game's Mods directory. Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File mod\build.ps1
# Pass -HKManaged for a non-default Steam library location.
param(
    [string]$HKManaged = "C:\Program Files (x86)\Steam\steamapps\common\Hollow Knight\hollow_knight_Data\Managed"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Without this guard the New-Item below would fabricate the whole tree and
# the install would "succeed" into a directory the game never reads.
if (-not (Test-Path $HKManaged)) {
    Write-Host "ERROR: Hollow Knight managed dir not found at:" -ForegroundColor Red
    Write-Host "  $HKManaged" -ForegroundColor Red
    exit 1
}

dotnet build -c Release -p:HKManaged="$HKManaged"
# $ErrorActionPreference does not cover native commands in Windows
# PowerShell 5.1, so the exit code must be checked by hand.
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

New-Item -ItemType Directory -Force -Path "$HKManaged\Mods\HKRLBot" | Out-Null
Copy-Item "bin\Release\net472\HKRLBot.dll" "$HKManaged\Mods\HKRLBot\"
Write-Host "HKRLBot installed to game Mods directory."
