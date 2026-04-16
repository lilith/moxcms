#!/usr/bin/env bash
# Unsafe-usage audit. Counts occurrences of the `unsafe` keyword in every
# `.rs` file under `src/` and compares against a pinned per-file baseline.
#
# The goal is monotonic reduction: any PR may remove `unsafe` or leave it
# unchanged; a PR that *increases* the count in a file (or introduces
# `unsafe` in a file that was previously clean) must bump the baseline
# deliberately — which is a visible review signal.
#
# Baseline file (TSV, sorted by file path):
#   <count>\t<path>
#   Only files with non-zero counts are listed; any non-listed file is
#   implicitly baselined at 0.
#
# Exit codes:
#   0 — every file is at or below its baseline count
#   1 — one or more files exceed their baseline
#   2 — infrastructure failure (missing baseline, etc.)
#
# Caveats: the count is grep-based (`\bunsafe\b`) and therefore includes
# occurrences inside string literals, doc comments, and `// unsafe` line
# comments. That's fine in practice — grep is deterministic, and the
# occasional noise just means the baseline gets bumped during legitimate
# doc / test edits, which is visible in review.

set -euo pipefail

die() { echo "unsafe-audit: $*" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BASELINE="$REPO_ROOT/ci/unsafe-audit/max-unsafe-per-file.tsv"
[ -f "$BASELINE" ] || die "missing $BASELINE"

OUTDIR="${OUTDIR:-$REPO_ROOT/target/unsafe-audit}"
mkdir -p "$OUTDIR"

CURRENT="$OUTDIR/current.tsv"
REGRESSIONS="$OUTDIR/regressions.tsv"

# Build current inventory: `<count>\t<path>` for every .rs under src/
# that contains at least one `unsafe` token.
while IFS= read -r -d '' path; do
  count=$(grep -cE '\bunsafe\b' "$path" || true)
  if [ "$count" -gt 0 ]; then
    printf '%s\t%s\n' "$count" "$path"
  fi
done < <(find src -name '*.rs' -type f -print0) \
  | sort -k2 > "$CURRENT"

# Ingest baseline into an associative array.
declare -A BASELINE_MAP
while IFS=$'\t' read -r count path; do
  [ -z "${path:-}" ] && continue
  case "$count" in \#*) continue ;; esac
  BASELINE_MAP[$path]=$count
done < "$BASELINE"

total_current=0
total_baseline=$(awk -F'\t' 'NR{ s += $1 } END { print s + 0 }' "$BASELINE")
regressions=0
: > "$REGRESSIONS"

while IFS=$'\t' read -r count path; do
  [ -z "${path:-}" ] && continue
  total_current=$((total_current + count))
  base=${BASELINE_MAP[$path]:-0}
  if [ "$count" -gt "$base" ]; then
    printf '%s\t%s\t%s\n' "$count" "$base" "$path" >> "$REGRESSIONS"
    regressions=$((regressions + 1))
  fi
done < "$CURRENT"

printf 'unsafe-audit: total `unsafe` tokens across src/: %s (baseline: %s, delta: %+d)\n' \
  "$total_current" "$total_baseline" "$((total_current - total_baseline))"

if [ "$regressions" -gt 0 ]; then
  echo "unsafe-audit: FAIL — $regressions file(s) exceed per-file baseline:"
  printf '              %4s %4s  %s\n' now base path
  while IFS=$'\t' read -r count base path; do
    printf '              %4s %4s  %s\n' "$count" "$base" "$path"
  done < "$REGRESSIONS"
  echo "              To accept the new counts intentionally, edit"
  echo "              $BASELINE"
  exit 1
fi

echo "unsafe-audit: result=PASS"
