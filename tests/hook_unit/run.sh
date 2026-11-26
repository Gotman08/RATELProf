#!/bin/bash
# usage: run.sh <ratelprof src dir>
SRC=$(realpath "${1:-$(dirname "$0")/../..}")
cd "$(dirname "$0")"
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
  FLAGSETS=("-O0" "-O2" "-O2 -mbranch-protection=standard" "-O3 -fPIC" "-Os")
else
  FLAGSETS=("-O0" "-O2" "-O2 -fcf-protection=full" "-O3 -fPIC" "-Os")
fi
gcc -O2 -fPIC -I$SRC/src/core/include -c $SRC/src/core/src/ratelprof/ratelprof_hook.c -o hook.o || exit 1
for pie in "-pie -fPIE" "-no-pie -fno-PIE"; do
 for f in "${FLAGSETS[@]}"; do
  gcc $f $pie -c funcs.c -o funcs.o || exit 1
  gcc -O1 $pie -I$SRC/src/core/include main.c funcs.o hook.o -o hook_test -lm || exit 1
  for mode in 0 1 2; do
    export RATELPROF_HOOK_FORCE_LONG=$mode
    out=$(./hook_test 2>&1); rc=$?
    printf "%-8s %-10s %-34s %-6s %s\n" "$ARCH" "${pie%% *}" "$f" "$mode" "$(echo "$out" | tail -1) (rc=$rc)"
    [ $rc -ne 0 ] && echo "$out" | grep -v "^PASSED\|^FAILED" | head -5
  done
 done
done
