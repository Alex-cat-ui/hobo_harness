#!/usr/bin/env bash
# Ollama lifecycle for development. The app owns this in production; this is
# for the terminal.
set -uo pipefail
export PATH="/opt/homebrew/bin:$PATH"
case "${1:-status}" in
  up)
    pgrep -f "ollama serve" >/dev/null && { echo "already up"; exit 0; }
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    for _ in $(seq 1 30); do
      curl -s --max-time 1 http://127.0.0.1:11434/api/tags >/dev/null && { echo "up"; exit 0; }
      sleep 1
    done
    echo "failed to come up in 30s; see /tmp/ollama.log"; exit 1 ;;
  down)
    pkill -f "ollama" 2>/dev/null
    sleep 2
    pgrep -f "ollama" >/dev/null && { echo "still running"; exit 1; } || echo "down" ;;
  status)
    curl -s --max-time 2 http://127.0.0.1:11434/api/ps >/dev/null && echo "up" || echo "down" ;;
  *) echo "usage: ollama.sh up|down|status"; exit 2 ;;
esac
