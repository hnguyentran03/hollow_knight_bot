#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
HK_MANAGED="$HOME/Library/Application Support/Steam/steamapps/common/Hollow Knight/hollow_knight.app/Contents/Resources/Data/Managed"

# Without this guard mkdir -p below would fabricate the whole tree and the
# install would "succeed" into a directory the game never reads.
if [ ! -d "$HK_MANAGED" ]; then
    echo "ERROR: Hollow Knight managed dir not found at:" >&2
    echo "  $HK_MANAGED" >&2
    exit 1
fi

./ensure_modding_api.sh "$HK_MANAGED"

dotnet build -c Release
mkdir -p "$HK_MANAGED/Mods/HKRLBot"
cp "bin/Release/net472/HKRLBot.dll" "$HK_MANAGED/Mods/HKRLBot/"
echo "HKRLBot installed to game Mods directory."
