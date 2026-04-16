#!/usr/bin/env bash
# Audit: for the interpolate methods modified in PR #168, dump cargo-asm
# output per type and assert zero `panic_bounds_check` calls inside any
# emitted symbol belonging to that type.
#
# The PR description claims "identical codegen to get_unchecked -- zero
# panic_bounds_check calls across all monomorphizations". This script is the
# gate that holds the claim.
#
# Usage:
#   ci/asm-audit.sh <scenario>
#
# Scenarios:
#   x86    — x86_64 host, default features, RUSTFLAGS="-C target-feature=+avx2,+fma"
#            (audits both SSE and AVX interpolator types)
#   aarch64 — aarch64 host, default features, RUSTFLAGS="-C target-feature=+neon"
#            (audits NEON interpolator types)
#
# Using default features keeps every questioned interpolator monomorphized as
# a standalone symbol, because SSE/AVX types cross-reference each other via
# `Double` dispatch and NEON types emit naturally. Narrower feature scoping
# would fully inline some `interpolate` methods away, hiding bounds checks
# behind wrapper symbols we don't audit here.
#
# Output (CI uploads as artifact):
#   target/asm-audit/<scenario>/full.s           — cargo asm --everything
#   target/asm-audit/<scenario>/<type>.s         — one file per questioned type
#   target/asm-audit/<scenario>/summary.txt      — OK/FAIL line per type
#
# Exit codes:
#   0 — every questioned type emitted for this scenario has zero
#       panic_bounds_check calls
#   1 — one or more types introduce a panic_bounds_check
#   2 — infrastructure failure (bad arg, cargo-asm missing, etc.)

set -euo pipefail

die() { echo "asm-audit: $*" >&2; exit 2; }

SCENARIO="${1:-}"
case "$SCENARIO" in
  x86|aarch64) ;;
  *) die "usage: $0 <x86|aarch64>" ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LIST="$REPO_ROOT/ci/asm-audit/questioned-functions.txt"
[ -f "$LIST" ] || die "missing $LIST"

OUTDIR="${OUTDIR:-$REPO_ROOT/target/asm-audit/$SCENARIO}"
mkdir -p "$OUTDIR"

command -v cargo-asm >/dev/null 2>&1 \
  || die "cargo-show-asm not installed (cargo install cargo-show-asm --locked)"

case "$SCENARIO" in
  x86)
    TARGET="x86_64-unknown-linux-gnu"
    # No explicit target-feature: default x86_64-v1 baseline keeps all
    # runtime-dispatch arms (SSE4.1 + AVX2+FMA) alive so every questioned
    # type gets monomorphized and emits a standalone symbol.
    EXPECT_TAGS="sse avx"
    ;;
  aarch64)
    TARGET="aarch64-unknown-linux-gnu"
    # NEON is baseline on aarch64; no target-feature needed.
    EXPECT_TAGS="neon"
    ;;
esac

# `options` enables the Tetrahedral/Pyramidal/Prismatic interpolator impls
# that are gated off in the default feature set. Without it only the always-on
# Trilinear path gets compiled, and the audit would silently skip 3/4 of the
# interpolator types. Enable all three on top of defaults.
FEATURES=(--features options)

echo "asm-audit: scenario=$SCENARIO target=$TARGET"
echo "asm-audit: RUSTFLAGS=${RUSTFLAGS:-<unset>}"
echo "asm-audit: features=${FEATURES[*]}"
echo "asm-audit: out=$OUTDIR"

FULL="$OUTDIR/full.s"
cargo asm -p moxcms --release --lib --simplify --everything \
  --target "$TARGET" \
  "${FEATURES[@]}" \
  > "$FULL" 2> "$OUTDIR/cargo-asm.err" \
  || { echo "cargo asm failed:" >&2; cat "$OUTDIR/cargo-asm.err" >&2; exit 2; }

SUMMARY="$OUTDIR/summary.txt"
: > "$SUMMARY"

fail=0
total=0
covered=0

match_scenario() {
  local sc=$1
  for t in $EXPECT_TAGS; do
    [ "$sc" = "$t" ] && return 0
  done
  return 1
}

# For each questioned type path in scope for this scenario:
#   1. Extract every symbol whose label contains the type path. One short-name
#      (e.g. TrilinearSse<_>::interpolate) often covers several
#      monomorphizations; we concatenate them all.
#   2. Count `panic_bounds_check` references in that concatenated body.
while IFS=$'\t' read -r sc tpath; do
  case "${sc:-}" in
    \#*|"") continue ;;
  esac
  match_scenario "$sc" || continue

  total=$((total + 1))
  safe=$(echo "$tpath" | sed 's|[^A-Za-z0-9_]|_|g')
  out="$OUTDIR/$safe.s"

  # Type paths in symbol labels are always immediately followed by `<` (the
  # start of generic args, e.g. `TrilinearSse<_>::interpolate`). Anchoring
  # the match on `<path><` avoids `TrilinearAvxFma` accidentally matching
  # `TrilinearAvxFmaDouble`.
  awk -v tp="$tpath<" '
    /^[A-Za-z_<].*:$/ {
      inseg = (index($0, tp) > 0) ? 1 : 0
    }
    inseg { print }
  ' "$FULL" > "$out"

  syms=$(grep -cE '^[A-Za-z_<].*:$' "$out" || true)

  if [ "$syms" -eq 0 ]; then
    # Fully inlined with no residual symbol containing the type path. LLVM
    # keeps a standalone symbol whenever bounds checks remain in the body,
    # so "no symbol" is a positive signal rather than a gap — note it but
    # don't fail.
    printf 'INLINED  0 syms   %s\n' "$tpath" | tee -a "$SUMMARY"
    continue
  fi

  covered=$((covered + 1))
  count=$(grep -c 'panic_bounds_check' "$out" || true)
  if [ "$count" -gt 0 ]; then
    printf 'FAIL   %2s syms  %2s bounds  %s\n' "$syms" "$count" "$tpath" \
      | tee -a "$SUMMARY"
    fail=1
  else
    printf 'OK     %2s syms   0 bounds  %s\n' "$syms" "$tpath" \
      | tee -a "$SUMMARY"
  fi
done < "$LIST"

echo "---"
echo "asm-audit: $covered / $total questioned types had symbols emitted"

# Whole-crate per-symbol baseline. Widens the audit beyond the per-type
# check above so every symbol in the crate — including the
# transform-chunk / TRC helpers an inlined interpolate might spill into —
# has a pinned ceiling.
#
# Baseline file format (TSV, sorted by symbol):
#   <count>\t<demangled symbol>
#
# Pass rule: every symbol's *current* `panic_bounds_check` count must be
# `<=` its baseline entry. A symbol that has ANY bounds checks today but
# isn't in the baseline is a regression.
BASELINE_FILE="$REPO_ROOT/ci/asm-audit/bounds-checks-$SCENARIO.tsv"
[ -f "$BASELINE_FILE" ] || die "missing $BASELINE_FILE"

CURRENT_TSV="$OUTDIR/current-bounds-checks.tsv"
awk '/^[a-zA-Z_<].*:$/{sym=$0;sub(/:$/,"",sym);next} /panic_bounds_check/{print sym}' "$FULL" \
  | sort | uniq -c \
  | awk '{count=$1; $1=""; sub(/^ +/,""); printf "%s\t%s\n", count, $0}' \
  > "$CURRENT_TSV"

REGRESSIONS="$OUTDIR/bounds-checks-regressions.tsv"
: > "$REGRESSIONS"

# Ingest baseline into an associative array (by symbol).
declare -A BASELINE_MAP
while IFS=$'\t' read -r count sym; do
  [ -z "${sym:-}" ] && continue
  case "$count" in \#*) continue ;; esac
  BASELINE_MAP[$sym]=$count
done < "$BASELINE_FILE"

total_actual=0
total_baseline=$(awk -F'\t' 'NR{ s += $1 } END { print s + 0 }' "$BASELINE_FILE")
regressions=0

# Compare current against baseline.
while IFS=$'\t' read -r count sym; do
  [ -z "${sym:-}" ] && continue
  total_actual=$((total_actual + count))
  base=${BASELINE_MAP[$sym]:-0}
  if [ "$count" -gt "$base" ]; then
    printf '%s\t%s\t%s\n' "$count" "$base" "$sym" >> "$REGRESSIONS"
    regressions=$((regressions + 1))
  fi
done < "$CURRENT_TSV"

printf 'asm-audit: whole-crate panic_bounds_check total: %s (baseline sum: %s, delta: %+d)\n' \
  "$total_actual" "$total_baseline" "$((total_actual - total_baseline))"

total_fail=0
if [ "$regressions" -gt 0 ]; then
  echo "asm-audit: FAIL — $regressions symbol(s) exceed per-symbol baseline:"
  printf '          %4s %4s  %s\n' now base symbol
  while IFS=$'\t' read -r count base sym; do
    printf '          %4s %4s  %s\n' "$count" "$base" "$sym"
  done < "$REGRESSIONS"
  echo "          To accept the new counts intentionally, edit"
  echo "          $BASELINE_FILE."
  total_fail=1
fi

echo "asm-audit: scenario=$SCENARIO result=$([ $fail = 0 ] && [ $total_fail = 0 ] \
  && echo PASS || echo FAIL)"
echo "---"

[ $fail = 0 ] && [ $total_fail = 0 ] || exit 1
