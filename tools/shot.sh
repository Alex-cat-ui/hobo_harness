#!/usr/bin/env bash
# Screenshot of the web interface, via headless Chrome.
#
# Deliberately not `screencapture`: that needs Screen Recording permission for
# the ssh session, and headless Chrome needs nothing at all. The interface is a
# web app, so this shows exactly what the Tauri window will show.
set -uo pipefail
URL="${1:-http://localhost:5173}"
OUT="${2:-/tmp/mshots/ui.png}"
SIZE="${3:-1600,1000}"
mkdir -p "$(dirname "$OUT")"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-sandbox --hide-scrollbars \
  --virtual-time-budget=4000 \
  --screenshot="$OUT" --window-size="$SIZE" "$URL" 2>/dev/null
[ -s "$OUT" ] && echo "$OUT ($(stat -f%z "$OUT") bytes)" || { echo "no screenshot produced"; exit 1; }
