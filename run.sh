#!/bin/bash
# One-command setup + launch (macOS). Safe to run every time:
#   bash run.sh [extra dashboard args, e.g. --port 9701]
# First run builds/installs the mod and creates the Python env; later runs
# skip whatever is already up to date and go straight to the dashboard.
# Windows users: run.ps1 is the same thing for PowerShell.
set -euo pipefail
cd "$(dirname "$0")"

HK_APP="$HOME/Library/Application Support/Steam/steamapps/common/Hollow Knight/hollow_knight.app"
HK_MANAGED="$HK_APP/Contents/Resources/Data/Managed"
INSTALLED_DLL="$HK_MANAGED/Mods/HKRLBot/HKRLBot.dll"
VENV_PY="trainer/.venv/bin/python"
REQ_STAMP="trainer/.venv/requirements-stamp"

# ---- figure out what this launch actually needs -----------------------------

need_venv=0
[ -x "$VENV_PY" ] || need_venv=1

need_pip=0
if [ "$need_venv" = 1 ] || ! cmp -s trainer/requirements.txt "$REQ_STAMP"; then
    need_pip=1
fi

# Rebuild when the installed dll is missing or any mod source is newer than it
# (generated sources under bin/ and obj/ don't count).
need_build=0
if [ ! -f "$INSTALLED_DLL" ]; then
    need_build=1
elif [ -n "$(find mod -path '*/bin' -prune -o -path '*/obj' -prune -o \
        \( -name '*.cs' -o -name '*.csproj' \) -newer "$INSTALLED_DLL" -print \
        2>/dev/null | head -1)" ]; then
    need_build=1
fi

# ---- dependency check: collect every problem, then report them together -----

missing=()

if [ ! -d "$HK_MANAGED" ]; then
    missing+=("Hollow Knight was not found at the default Steam location:
      $HK_APP
    Install it via Steam, or if it lives in a different Steam library,
    edit HK_APP at the top of run.sh (and HK_MANAGED in mod/build.sh).")
fi

if [ "$need_venv" = 1 ]; then
    if ! command -v python3 >/dev/null 2>&1; then
        missing+=("Python 3.11+ is required but python3 was not found.
    Install it with 'brew install python' or from https://www.python.org/downloads/")
    elif ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
        missing+=("Python 3.11+ is required but python3 is $(python3 -V 2>&1 | awk '{print $2}').
    Install a newer one with 'brew install python' or from https://www.python.org/downloads/")
    fi
fi

if [ "$need_build" = 1 ]; then
    if ! command -v dotnet >/dev/null 2>&1; then
        missing+=(".NET SDK is required to build the mod but 'dotnet' was not found.
    Install it with 'brew install dotnet-sdk' or from https://dotnet.microsoft.com/download")
    fi
    # The Modding API lives inside the patched Assembly-CSharp.dll ("ModHooks"
    # appears only in the patched assembly). A vanilla active dll is fine if
    # Lumafly's modded backup exists -- mod/build.sh swaps it back in itself.
    if [ -d "$HK_MANAGED" ] \
        && ! grep -aq ModHooks "$HK_MANAGED/Assembly-CSharp.dll" 2>/dev/null \
        && ! grep -aq ModHooks "$HK_MANAGED/Assembly-CSharp.dll.m" 2>/dev/null; then
        missing+=("The Hollow Knight Modding API is not installed (Assembly-CSharp.dll is vanilla).
    Install it with Lumafly (https://themulhima.github.io/Lumafly/) and run this script again.")
    fi
fi

if [ "${#missing[@]}" -gt 0 ]; then
    echo ""
    echo "Cannot start yet -- ${#missing[@]} thing(s) need attention:" >&2
    n=0
    for m in "${missing[@]}"; do
        n=$((n + 1))
        echo "" >&2
        echo "  $n. $m" >&2
    done
    echo "" >&2
    exit 1
fi

# ---- setup steps (each skipped when already done) ---------------------------

if [ "$need_venv" = 1 ]; then
    echo "==> Creating Python environment (first run only)..."
    python3 -m venv trainer/.venv
fi

if [ "$need_pip" = 1 ]; then
    echo "==> Installing Python dependencies (takes a few minutes the first time)..."
    "$VENV_PY" -m pip install -q -r trainer/requirements.txt
    cp trainer/requirements.txt "$REQ_STAMP"
fi

if [ "$need_build" = 1 ]; then
    echo "==> Building and installing the HKRLBot mod..."
    bash mod/build.sh
    # Copying the dll into the app bundle invalidates its code signature and
    # unsigned launches die instantly (exit 138), so re-sign after every build.
    echo "==> Re-signing the game app..."
    codesign --force --deep --sign - "$HK_APP"
fi

# ---- launch -----------------------------------------------------------------

echo "==> Starting the dashboard (Ctrl-C stops it; training runs it launched keep going)..."
cd trainer
exec ./.venv/bin/python scripts/dashboard.py --open "$@"
