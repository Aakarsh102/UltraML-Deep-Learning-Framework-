#!/usr/bin/env bash
# Type-checks every UltraML source file with plain clang++ on machines that
# have no CUDA toolkit (e.g. a Mac laptop). It mirrors the tree into a temp
# dir, strips the <<<grid, block>>> launch syntax so kernels parse as normal
# functions, and compiles each translation unit against the stub CUDA /
# cuBLAS / cuDNN headers in stubs/.
#
# This catches interface mismatches, template errors, and most type bugs —
# it does NOT execute anything or validate numerics; run the examples on a
# real GPU for that.
#
# Usage:  tools/host_check/check.sh

set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

cp -R "$ROOT/". "$WORK/"
rm -rf "$WORK/build" "$WORK/tools"

find "$WORK" -name '*.cu' | while read -r f; do
    perl -pe 's/<<<[^>]*>>>//g' "$f" > "${f%.cu}_cu.cpp"
done

fail=0
for f in $(find "$WORK" -name '*_cu.cpp' | sort) "$WORK"/examples/*.cpp; do
    rel="${f#$WORK/}"
    # -Wno-undefined-internal: `extern __shared__` arrays lose their storage
    # qualifier under the stubs; harmless for a syntax-only pass.
    if clang++ -std=c++17 -fsyntax-only -Wno-undefined-internal \
               -I "$HERE/stubs" "$f"; then
        echo "OK:   $rel"
    else
        echo "FAIL: $rel"
        fail=1
    fi
done
exit $fail
