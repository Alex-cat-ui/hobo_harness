#!/usr/bin/env bash
# One-shot environment report. First thing to run when something is wrong.
set -uo pipefail
export PATH="$HOME/.cargo/bin:/opt/homebrew/bin:$PATH"
cd "$(dirname "$0")/.."

line() { printf '%-26s %s\n' "$1" "$2"; }
echo "=== toolchain ==="
line "rustc"  "$(rustc --version 2>/dev/null || echo MISSING)"
line "cargo"  "$(cargo --version 2>/dev/null || echo MISSING)"
line "node"   "$(node --version 2>/dev/null || echo MISSING)"
line "npm"    "$(npm --version 2>/dev/null || echo MISSING)"
line "git"    "$(git --version 2>/dev/null || echo MISSING)"
line "ollama" "$(ollama --version 2>&1 | grep -o 'version is .*' || echo MISSING)"

echo
echo "=== ollama ==="
if curl -s --max-time 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  line "server" "up"
  curl -s http://127.0.0.1:11434/api/tags | python3 -c '
import json,sys
for m in json.load(sys.stdin).get("models",[]):
    print("  {:<26} {:>6.2f} GiB".format(m["name"], m["size"]/2**30))'
  echo "  resident now:"
  curl -s http://127.0.0.1:11434/api/ps | python3 -c '
import json,sys
ms=json.load(sys.stdin).get("models",[])
if not ms:
    print("    (none)")
for m in ms:
    print("    {:<24} {:>6.2f} GiB".format(m["name"], m["size"]/2**30))'
else
  line "server" "DOWN — start with tools/ollama.sh up"
fi

echo
echo "=== memory ==="
vm_stat | awk -v ps=16384 '
/page size of/ { match($0,/[0-9]+/); ps=substr($0,RSTART,RLENGTH) }
/Pages free/ {gsub(/\./,"");f=$3}
/Pages inactive/ {gsub(/\./,"");i=$3}
/Pages speculative/ {gsub(/\./,"");s=$3}
/Pages purgeable/ {gsub(/\./,"");p=$3}
END { printf "  available %.2f GiB\n", (f+i+s+p)*ps/2^30 }'

echo
echo "=== repository ==="
line "branch" "$(git branch --show-current 2>/dev/null)"
line "head"   "$(git log --oneline -1 2>/dev/null)"
line "dirty"  "$(git status --porcelain | wc -l | tr -d ' ') files"

echo
echo "=== build ==="
cargo build --release 2>&1 | tail -1
echo "=== tests ==="
cargo test --release 2>&1 | grep -E "^test result: ok" | awk '{s+=$4} END {print "  passed:", s}'
cargo test --release 2>&1 | grep -cE "FAILED" | awk '{print "  failed:", $1}'
