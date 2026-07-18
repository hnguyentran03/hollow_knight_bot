#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
HK_MANAGED="$HOME/Library/Application Support/Steam/steamapps/common/Hollow Knight/hollow_knight.app/Contents/Resources/Data/Managed"
dotnet build -c Release
mkdir -p "$HK_MANAGED/Mods/HKRLBot"
cp "bin/Release/net472/HKRLBot.dll" "$HK_MANAGED/Mods/HKRLBot/"
echo "HKRLBot installed to game Mods directory."
