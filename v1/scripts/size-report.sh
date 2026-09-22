#!/usr/bin/env bash
# Size budget check (design doc 16.3): fail the build on a regression.
#
#   ./scripts/size-report.sh --fail-over simd128=48000 relaxed=56000
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
declare -A LIMIT
mode=""
for arg in "$@"; do
  case "$arg" in
    --fail-over) mode=limit ;;
    *=*) [ "$mode" = limit ] && LIMIT["${arg%%=*}"]="${arg##*=}" ;;
  esac
done

status=0
printf "%-10s %10s %10s %10s\n" build raw gzip budget
for build in scalar simd128 relaxed; do
  f="$ROOT/dist/lnn-$build.wasm"
  [ -f "$f" ] || continue
  raw=$(stat -c%s "$f")
  gz=$(gzip -9 -c "$f" | wc -c)
  limit="${LIMIT[$build]:-}"
  printf "%-10s %10d %10d %10s" "$build" "$raw" "$gz" "${limit:--}"
  if [ -n "$limit" ] && [ "$gz" -gt "$limit" ]; then
    printf "  OVER BUDGET by %d bytes\n" "$((gz - limit))"
    status=1
  else
    printf "\n"
  fi
done
exit $status
