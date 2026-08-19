#!/usr/bin/env bash
# Everything that must be true before a commit, in one command.
#
# The check-list in docs/DEVELOPING.md used to be prose, which meant it was
# followed from memory. This is the same list, executable.
#
#   ./tools/check.sh            build, tests, warnings, document links
#   ./tools/check.sh --mutants  the above plus mutation runs on the two modules
#                               that carry the safety guarantees
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.cargo/bin:/opt/homebrew/bin:$PATH"

fail=0
step() { printf "\n=== %s\n" "$1"; }
bad()  { printf "  FAIL  %s\n" "$1"; fail=1; }
ok()   { printf "  ok    %s\n" "$1"; }

step "build"
if out=$(cargo build --release 2>&1); then ok "release build"; else
  echo "$out" | grep -E "^error" | head -20; bad "build"; fi

step "warnings"
# Cargo only re-reports warnings for what it actually recompiled, so a cached
# build says nothing. That is reported rather than counted as a pass.
known=2   # backend.rs:9 and run.rs:18 — legacy of the line-syntax path, task T-056
if ! printf "%s" "$out" | grep -q "Compiling"; then
  ok "build was cached, warnings not re-reported (touch a source to see them)"
else
  warnings=$(printf "%s" "$out" | grep -E "^warning:" | grep -vc "generated")
  if [ "$warnings" -le "$known" ]; then ok "$warnings warning(s), $known known, none new"; else
    printf "%s" "$out" | grep -A1 "^warning:" | grep -- "-->"; bad "$warnings warnings, $known known"; fi
fi

step "tests"
if out=$(cargo test --release 2>&1); then
  total=$(printf "%s" "$out" | grep -oE "^test result: ok\. [0-9]+ passed" | awk '{s+=$4} END {print s}')
  ok "${total:-0} tests, none failed"
else
  printf "%s" "$out" | grep -E "^(test .* FAILED|failures:|error)" | head -20; bad "tests"; fi

step "document links"
if command -v python3 >/dev/null; then
  python3 - <<'PY' || fail=1
import os, re, sys
bad = []
for root, _, files in os.walk("docs"):
    for f in files:
        if not f.endswith(".md"):
            continue
        p = os.path.join(root, f)
        for link in re.findall(r"\]\((\./[^)#]+|\.\./[^)#]+)\)", open(p).read()):
            if not os.path.exists(os.path.normpath(os.path.join(root, link))):
                bad.append(f"{p} -> {link}")
print(f"  ok    {len(bad)} broken link(s)" if not bad else "  FAIL  broken links:")
for b in bad:
    print("       ", b)
sys.exit(1 if bad else 0)
PY
else
  bad "python3 is needed for the link check"
fi

if [ "${1:-}" = "--mutants" ]; then
  for m in sandbox journal; do
    step "mutants: $m"
    if cargo mutants --file "crates/minions-core/src/$m.rs" 2>&1 | tail -3; then
      ok "$m"; else bad "mutants on $m"; fi
  done
fi

printf "\n"
[ "$fail" -eq 0 ] && echo "ALL CLEAR" || echo "SOMETHING IS NOT TRUE — see FAIL above"
exit "$fail"
