#!/usr/bin/env bash
# build.sh — compile the PATCHED vllm_exl3_c extension (multi-K p2b MoE,
# kernel-work/patched/) for SM121 (GB10, compute_121) on a build node
# (201-204) inside the deepseek-v41-exl3:fresh image.
#
# Phases:
#   probe   record the K5/K6 decoder-route evidence (dq8_regs_* / dq_dispatch
#           greps) and diff the node's /opt/vllm-exl3/csrc against the staged
#           kernel-work/csrc pin into kernel-work/patched/build-probe.log;
#   overlay copy /opt/vllm-exl3 to a scratch tree and overlay the three
#           patched sources (the installed plugin is never mutated);
#   build   mode A: the proven pip recipe (setup.py build_ext --inplace with
#               EXL3_EXT_INCLUDE + TORCH_CUDA_ARCH_LIST=12.1a + NVCC_APPEND_FLAGS
#               -Xptxas -v);
#           mode B: torch.utils.cpp_extension driver invoking nvcc directly with
#               -gencode arch=compute_121,code=sm_121 -Xptxas -v (fallback);
#   smoke   import the produced .so and assert ABI 4 + the new and old
#           entry points.
#
# The result lands in kernel-work/patched/build/vllm_exl3_c*.so together with
# the ptxas -v log. DEPLOYMENT (docker cp into dsv41-tp4, serve restart) is
# explicitly OUT OF SCOPE — this script stages only.
#
# Usage: bash kernel-work/build.sh [--probe-only] [--mode a|b] [--skip-smoke]
#                                 [--src-tree /opt/vllm-exl3] [--allow-drift]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHED="$SCRIPT_DIR/patched"
STAGED="$SCRIPT_DIR/csrc"
OUT="$PATCHED/build"
PROBE_LOG="$PATCHED/build-probe.log"

MODE="a"
PROBE_ONLY=0
SKIP_SMOKE=0
ALLOW_DRIFT=0
SRC_TREE="/opt/vllm-exl3"
while [ $# -gt 0 ]; do
    case "$1" in
        --probe-only) PROBE_ONLY=1 ;;
        --mode) MODE="$2"; shift ;;
        --skip-smoke) SKIP_SMOKE=1 ;;
        --src-tree) SRC_TREE="$2"; shift ;;
        --allow-drift) ALLOW_DRIFT=1 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

for f in "$PATCHED/p2b_moe.cu" "$PATCHED/p2b_moe.cuh" "$PATCHED/bindings.cpp"; do
    [ -f "$f" ] || { echo "missing patched source: $f" >&2; exit 1; }
done

# ---------------------------------------------------------------------------
# exllamav3 extension headers (util.h/util.cuh, quant/exl3_gemv_kernel.cuh,
# quant/exl3_dq.cuh) — the same include tree setup.py consumes.
# ---------------------------------------------------------------------------
EXT_INC="${EXL3_EXT_INCLUDE:-}"
if [ -z "$EXT_INC" ] || [ ! -d "$EXT_INC" ]; then
    for c in /opt/exllamav3/exllamav3/exllamav3_ext; do
        [ -d "$c" ] && EXT_INC="$c" && break
    done
fi
if [ -z "$EXT_INC" ] || [ ! -d "$EXT_INC" ]; then
    EXT_INC="$(python3 - <<'PY'
try:
    import exllamav3, os
    p = os.path.join(os.path.dirname(exllamav3.__file__), "exllamav3_ext")
    print(p if os.path.isdir(p) else "")
except Exception:
    print("")
PY
)"
fi
[ -n "$EXT_INC" ] && [ -d "$EXT_INC" ] \
    || { echo "exllamav3_ext include tree not found (set EXL3_EXT_INCLUDE)" >&2; exit 1; }
echo "EXL3_EXT_INCLUDE=$EXT_INC"

# ---------------------------------------------------------------------------
# Probe: K5/K6 decoder route evidence + node-source drift check.
# Route (b) (dq_dispatch, K1..8) is the default compiled into the patched
# kernel; if the node ships dq8_regs_5bits/6bits instead, route (a) becomes an
# option — the parity gate does not care which, but the probe log must be
# recorded before the kernel route is considered frozen.
# ---------------------------------------------------------------------------
mkdir -p "$OUT"
{
    echo "# build.sh probe  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "EXL3_EXT_INCLUDE=$EXT_INC"
    echo
    echo "## dq8_regs_* inventory (register decoders; K5/K6 presence would enable route a)"
    grep -n "dq8_regs_" "$EXT_INC/quant/exl3_gemv_kernel.cuh" 2>&1 | head -40 || true
    echo
    echo "## dq_dispatch signature evidence (route b; covers K1..8)"
    grep -n "dq_dispatch\|template <int bits" "$EXT_INC/quant/exl3_dq.cuh" 2>&1 | head -20 || true
    echo
    echo "## node csrc vs staged pin ($SRC_TREE/csrc vs $STAGED)"
    if [ -d "$SRC_TREE/csrc" ]; then
        if diff -r "$SRC_TREE/csrc" "$STAGED" > "$OUT/probe-csrc.diff" 2>&1; then
            echo "IDENTICAL (patched files anchor cleanly)"
        else
            echo "DRIFT — see $OUT/probe-csrc.diff"
        fi
    else
        echo "MISSING: $SRC_TREE/csrc"
    fi
    echo
    echo "## nvidia compiler"
    nvcc --version 2>&1 | tail -2 || true
} | tee "$PROBE_LOG"

if diff -r "$SRC_TREE/csrc" "$STAGED" >/dev/null 2>&1 || [ ! -d "$SRC_TREE/csrc" ]; then
    :
else
    if [ "$ALLOW_DRIFT" -eq 0 ]; then
        echo "" >&2
        echo "FATAL: the node's $SRC_TREE/csrc differs from the staged pin." >&2
        echo "Re-anchor kernel-work/patched to the node copy before building" >&2
        echo "(see $OUT/probe-csrc.diff), or pass --allow-drift to build anyway." >&2
        exit 1
    fi
    echo "WARNING: building over drifted sources (--allow-drift)" >&2
fi

[ "$PROBE_ONLY" -eq 0 ] || { echo "probe-only: done ($PROBE_LOG)"; exit 0; }

# ---------------------------------------------------------------------------
# Overlay: scratch copy of the plugin tree + the three patched sources.
# ---------------------------------------------------------------------------
WORK="$(mktemp -d /tmp/vllm-exl3-mkbuild.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
cp -a "$SRC_TREE" "$WORK/src"
cp "$PATCHED/p2b_moe.cu" "$PATCHED/p2b_moe.cuh" "$PATCHED/bindings.cpp" "$WORK/src/csrc/"
echo "overlay tree: $WORK/src"

# ---------------------------------------------------------------------------
# Build.
# ---------------------------------------------------------------------------
build_mode_a() {
    cd "$WORK/src"
    export EXL3_EXT_INCLUDE="$EXT_INC"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
    export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:--Xptxas -v}"
    export MAX_JOBS="${MAX_JOBS:-8}"
    echo "mode A: setup.py build_ext --inplace (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST)"
    python3 setup.py build_ext --inplace 2>&1 | tee "$OUT/build-mode-a.log"
    local so
    so="$(find "$WORK/src" -maxdepth 2 -name 'vllm_exl3_c*.so' | head -1 || true)"
    [ -n "$so" ] || { echo "mode A produced no .so" >&2; return 1; }
    cp "$so" "$OUT/"
    echo "mode A ok: $OUT/$(basename "$so")"
}

build_mode_b() {
    echo "mode B: torch.utils.cpp_extension (nvcc -gencode arch=compute_121,code=sm_121 -Xptxas -v)"
    local bdir="$OUT/mode-b"
    mkdir -p "$bdir"
    python3 - "$WORK/src/csrc" "$EXT_INC" "$bdir" "$OUT" <<'PY' 2>&1 | tee "$OUT/build-mode-b.log"
import os, sys, shutil
import torch
from torch.utils.cpp_extension import load

csrc, ext_inc, build_dir, out_dir = sys.argv[1:5]
sources = [os.path.join(csrc, s) for s in (
    "bindings.cpp",
    "exl3_gemv.cu",
    "p2b_batched.cu",
    "p2b_moe.cu",
    "exl3_gemm.cu",
    "exl3_fat_gemm.cu",
)]
for s in sources:
    assert os.path.isfile(s), s

module = load(
    name="vllm_exl3_c",
    sources=sources,
    extra_include_paths=[csrc, ext_inc, os.path.join(ext_inc, "quant")],
    extra_cuda_cflags=[
        "-O3", "-std=c++17",
        "-gencode", "arch=compute_121,code=sm_121",
        "-Xptxas", "-v",
    ],
    extra_cflags=["-O3", "-std=c++17"],
    build_directory=build_dir,
    verbose=True,
)
so = module.__file__
assert so and os.path.isfile(so), so
dst = os.path.join(out_dir, os.path.basename(so))
shutil.copy2(so, dst)
print("mode B ok:", dst)
PY
    ls "$OUT"/vllm_exl3_c*.so >/dev/null 2>&1 || { echo "mode B produced no .so in $OUT" >&2; return 1; }
}

case "$MODE" in
    a) build_mode_a ;;
    b) build_mode_b ;;
    trya)
        build_mode_a || { echo "mode A failed; falling back to mode B" >&2; build_mode_b; }
        ;;
    *) echo "unknown --mode $MODE (a | b | trya)" >&2; exit 2 ;;
esac
MODE_USED="$MODE"

# Save the ptxas -v evidence with the artifact.
case "$MODE_USED" in
    a) cp "$OUT/build-mode-a.log" "$OUT/build-ptxas.log" ;;
    b|trya) cp "$OUT/build-mode-b.log" "$OUT/build-ptxas.log" ;;
esac
if grep -q "p2b_moe_mixedk_kernel" "$OUT/build-ptxas.log" 2>/dev/null; then
    echo "ptxas -v: entries for p2b_moe_mixedk_kernel recorded in $OUT/build-ptxas.log"
    grep -A3 "p2b_moe_mixedk_kernel" "$OUT/build-ptxas.log" | head -40 || true
else
    echo "ptxas -v: no p2b_moe_mixedk_kernel entry found in build-ptxas.log (inspect manually)" >&2
fi

# ---------------------------------------------------------------------------
# Smoke: import the staged .so; ABI 4 + new and old entries.
# ---------------------------------------------------------------------------
if [ "$SKIP_SMOKE" -eq 0 ]; then
    SO="$(ls "$OUT"/vllm_exl3_c*.so | head -1)"
    [ -n "$SO" ] || { echo "no .so to smoke-test" >&2; exit 1; }
    SO="$SO" python3 - <<'PY'
import importlib.util, os, sys

so = os.environ["SO"]
import torch  # extension links against torch; import it first
spec = importlib.util.spec_from_file_location("vllm_exl3_c", so)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

abi = getattr(mod, "P2B_MOE_ABI_VERSION", None)
assert abi == 4, f"P2B_MOE_ABI_VERSION == {abi}, expected 4"
assert bool(getattr(mod, "P2B_MOE_MIXED_K", False)), "P2B_MOE_MIXED_K missing/false"
for entry in ("p2b_fused_moe_mk", "p2b_fused_moe", "exl3_gemv", "p2b_gemv_batched"):
    assert callable(getattr(mod, entry, None)), f"missing entry: {entry}"
print(f"smoke: OK  {so}")
print(f"smoke: ABI 4, P2B_MOE_MIXED_K=true, entries: p2b_fused_moe_mk + p2b_fused_moe/exl3_gemv/p2b_gemv_batched")
PY
fi

echo "build.sh: DONE (staged under $OUT — deployment to dsv41-tp4 is out of scope)"
