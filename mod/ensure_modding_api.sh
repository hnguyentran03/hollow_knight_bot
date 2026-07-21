#!/bin/bash
# Detect a Steam-reverted vanilla Assembly-CSharp.dll and swap Lumafly's
# modded backup back in. The Hollow Knight Modding API lives INSIDE the
# patched Assembly-CSharp.dll (there is no separate Modding.dll), and a
# Steam file re-validation can silently restore vanilla -- builds then die
# with CS0246 'Modding' not found and the game loads no mods (bit us
# 2026-07-21). Detection is by content: the type name "ModHooks" appears in
# the patched assembly's metadata and never in vanilla (verified against
# the real dlls: grep -ac ModHooks = 2 modded, 0 vanilla). Repairs only the
# one provably-safe case; never guesses, downloads, or edits bytes.
set -euo pipefail

managed="${1:?usage: ensure_modding_api.sh <managed-dir>}"
active="$managed/Assembly-CSharp.dll"
modded="$managed/Assembly-CSharp.dll.m"    # Lumafly convention: modded backup
vanilla="$managed/Assembly-CSharp.dll.v"   # Lumafly convention: vanilla backup

has_api() { grep -aq ModHooks "$1"; }      # -a: treat the binary as text

if [ ! -f "$active" ]; then
    echo "ERROR: $active is missing entirely." >&2
    echo "  This is a broken install, not a Steam revert; repair it with Lumafly" >&2
    echo "  (or Steam 'verify integrity' first, then Lumafly), then rebuild." >&2
    exit 1
fi

if has_api "$active"; then
    exit 0    # healthy modded install: stay quiet
fi

if [ -f "$modded" ] && has_api "$modded"; then
    cp -p "$active" "$vanilla"
    cp -p "$modded" "$active"
    echo "Modding API restored: Assembly-CSharp.dll was vanilla (Steam re-validation?);" \
         "swapped Assembly-CSharp.dll.m back in, vanilla kept as Assembly-CSharp.dll.v."
    exit 0
fi

echo "ERROR: Assembly-CSharp.dll is vanilla (no Modding API) and there is no usable" >&2
echo "  modded backup to restore:" >&2
if [ -f "$modded" ]; then
    echo "  $modded exists but is also vanilla." >&2
else
    echo "  $modded is missing." >&2
fi
echo "  Reinstall the Modding API with Lumafly, then rebuild." >&2
exit 1
