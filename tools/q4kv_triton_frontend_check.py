"""Frontend (AST -> TTIR -> codegen) compile check for the q4 KV Triton
kernels, for the GB10 target (sm_121), with NO GPU required.

``tools/gb10_q4kv_parity.py`` runs this FIRST so that a frontend type
error (the GB10 val1 failure class: a raw fp8 pointer handed to a
kernel that stores raw uint8 bytes) fails fast with a clear message
instead of dying inside case 1.

What it does:
- imports the real W1/R2 kernels from ``vllm_exl3.minimax_q4kv_kernels``
  (the same objects the wrappers launch);
- compiles each kernel with ``triton.compile(ASTSource(...),
  target=GPUTarget("cuda", 121, 32))`` for a matrix of host-side dtype
  variants (the parity harness's int64 index tensors AND vLLM's int32
  serving tensors; fp32 and bf16 norm/cos_sin weights);
- enforces the q4 boundary convention: the fp8 e4m3 index buffers must
  appear in every signature as uint8 byte views (``*u8``), never as fp8
  pointers, and no signature may contain an fp8 pointer type at all.

Exit status: 0 iff every variant compiles; non-zero (with the Triton
error text) otherwise. Requires the GB10 vLLM image (triton + ptxas);
it does NOT require a GPU, CUDA driver, or a writable repo checkout.

This VPS has no triton: the tool has never been executed here. Its
first real execution must be on GB10, via the parity harness.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vllm_exl3 import minimax_q4kv_kernels as q4k  # noqa: E402


def _fail(msg: str) -> None:
    print(f"FRONTEND CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# signature construction
# ---------------------------------------------------------------------------


def _param_names(fn) -> list[str]:
    if hasattr(fn, "params"):
        names = [p.name for p in fn.params]
    else:  # older triton
        names = list(fn.arg_names)
    if not names:
        _fail(f"cannot enumerate the parameters of {fn}")
    return names


def _constexpr_names(fn) -> set[str]:
    out = set()
    params = getattr(fn, "params", None)
    if params is not None:
        for p in params:
            if getattr(p, "is_constexpr", False):
                out.add(p.name)
    return out


def _make_source(triton, fn, variant: dict):
    """Build an ASTSource for `fn` under the dtype `variant`."""
    ASTSource = triton.compiler.ASTSource

    names = _param_names(fn)
    cnames = _constexpr_names(fn)
    signature: dict[str, str] = {}
    constants: dict[str, object] = {}
    for name in names:
        if name in cnames:
            signature[name] = "constexpr"
            if name not in variant:
                _fail(f"variant is missing constexpr {name}")
            constants[name] = variant[name]
        elif name not in variant:
            _fail(f"variant is missing parameter {name}")
        else:
            signature[name] = variant[name]
    # The q4 boundary convention: fp8 e4m3 buffers cross as uint8 byte
    # views; an fp8 pointer anywhere in the signature is a regression.
    for name, ty in signature.items():
        if "fp8" in ty:
            _fail(f"{fn} parameter {name} is typed {ty}: fp8 buffers "
                  "must be handed to the q4 kernels as uint8 byte views")
    params = inspect.signature(ASTSource.__init__).parameters
    kwargs = {}
    if "constexprs" in params:
        kwargs["constexprs"] = constants
    elif "constants" in params:
        kwargs["constants"] = constants
    else:
        _fail("unsupported triton ASTSource API "
              f"({', '.join(params)})")
    return ASTSource(fn=fn, signature=signature, **kwargs)


def _compile_variant(triton, fn, label: str, variant: dict) -> None:
    GPUTarget = triton.backends.compiler.GPUTarget

    src = _make_source(triton, fn, variant)
    target = GPUTarget("cuda", 121, 32)
    compiled = None
    try:
        compiled = triton.compile(src, target=target)
    except Exception as exc:  # CompilationError and friends
        print(f"[{label}] COMPILE FAILED:", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        _fail(f"{label} does not frontend-compile for sm_121")
    asm = getattr(compiled, "asm", {}) or {}
    ttir = asm.get("ttir", "")
    n_lines = len(ttir.splitlines())
    if n_lines == 0:
        _fail(f"{label} compiled but produced no TTIR")
    print(f"[{label}] OK: sm_121 TTIR generated ({n_lines} lines)")


# ---------------------------------------------------------------------------
# the kernels and their host-side dtype variants
# ---------------------------------------------------------------------------


def _w1_variants() -> dict[str, dict]:
    # Pointer order of _q4_w1_kernel (pinned by q4_w1_insert):
    # qkv, q_out, index_q_out, q_norm_w, k_norm_w, iq_norm_w, ik_norm_w,
    # cos_sin, pos, slot_map, index_slot_map, pool, index_cache, ...
    base = {
        "qkv_ptr": "*bf16",
        "q_out_ptr": "*bf16",
        # fp8 e4m3 buffers, always as uint8 byte views (_fp8_bytes).
        "index_q_out_ptr": "*u8",
        "index_cache_ptr": "*u8",
        "pool_ptr": "*u8",
        "eps": "fp32",
        "rotary_dim": "i32",
        "num_tokens": "i32",
        "nq": "i32",
        "nkv": "i32",
        "niq": "i32",
        "qkv_row": "i32",
        "pool_page_stride": "i32",
    }
    harness = {
        **base,
        # tools/gb10_q4kv_parity.py case 1 (torch defaults are int64).
        "q_norm_w_ptr": "*fp32",
        "k_norm_w_ptr": "*fp32",
        "iq_norm_w_ptr": "*fp32",
        "ik_norm_w_ptr": "*fp32",
        "cos_sin_ptr": "*fp32",
        "pos_ptr": "*i64",
        "slot_map_ptr": "*i64",
        "index_slot_map_ptr": "*i64",
        "PROCESS_INDEX": True,
    }
    serving_i32 = {
        **base,
        # vLLM serving tensors: int32 index maps, bf16 weights.
        "q_norm_w_ptr": "*bf16",
        "k_norm_w_ptr": "*bf16",
        "iq_norm_w_ptr": "*bf16",
        "ik_norm_w_ptr": "*bf16",
        "cos_sin_ptr": "*bf16",
        "pos_ptr": "*i64",
        "slot_map_ptr": "*i32",
        "index_slot_map_ptr": "*i32",
        "PROCESS_INDEX": True,
    }
    skip_index = {
        **harness,
        # skip_index_branch=True route (index code pruned).
        "PROCESS_INDEX": False,
    }
    return {
        "W1/parity-harness": harness,
        "W1/serving-i32": serving_i32,
        "W1/skip-index": skip_index,
    }


def _r2_variants() -> dict[str, dict]:
    # Pointer order of _q4_decode_kernel (pinned by q4_sparse_decode).
    base = {
        "q_ptr": "*bf16",
        "pool_ptr": "*u8",
        "o_ptr": "*bf16",
        "lse_ptr": "*fp32",
        "total_q": "i32",
        "gqa_group_size": "i32",
        "max_topk": "i32",
        "sm_scale": "fp32",
        "decode_query_len": "i32",
        "stride_qn": "i32",
        "stride_qh": "i32",
        "stride_qd": "i32",
        "pool_page_stride": "i32",
        "stride_th": "i32",
        "stride_tn": "i32",
        "stride_tk": "i32",
        "stride_o_c": "i32",
        "stride_o_b": "i32",
        "stride_o_h": "i32",
        "stride_o_d": "i32",
        "stride_l_c": "i32",
        "stride_l_b": "i32",
        "stride_l_h": "i32",
        "stride_bt_b": "i32",
        # The wrapper's shape-constant chunk count (gqa 16 -> BLOCK_H 16).
        "NUM_TOPK_CHUNKS": 8,
        "BLOCK_H": 16,
    }
    return {
        # tools/gb10_q4kv_parity.py case 2 (torch defaults are int64).
        "R2/parity-harness": {
            **base,
            "t_ptr": "*i64",
            "block_table_ptr": "*i64",
            "seq_lens": "*i64",
        },
        # vLLM serving tensors are int32.
        "R2/serving-i32": {
            **base,
            "t_ptr": "*i32",
            "block_table_ptr": "*i32",
            "seq_lens": "*i32",
        },
    }


def _stage_variants() -> dict[str, dict]:
    # Pointer order of _q4_stage_kernel (pinned by q4_stage_pages):
    # pool_ptr, stage_ptr, pages_ptr, n_heads, BLOCK_T (constexpr).
    base = {
        "pool_ptr": "*u8",
        "stage_ptr": "*bf16",
        "pages_ptr": "*i64",
        "n_heads": "i32",
        "BLOCK_T": 32,
    }
    return {
        # tools/gb10_q4kv_parity.py units 6/7 (torch defaults are int64;
        # q4_stage_pages normalizes to int64 before the launch).
        "R1stage/parity-harness": dict(base),
        # int32 page ids are accepted by the wrapper and cast on device.
        "R1stage/serving-i32": {**base, "pages_ptr": "*i32"},
    }


def main() -> None:
    print("q4 KV Triton frontend check (sm_121, no GPU needed)")
    try:
        tl, triton = q4k._import_triton()  # noqa: F841
    except Exception as exc:
        _fail(f"triton is not importable ({exc}); the q4 kernels cannot "
              "be frontend-checked in this environment")
    w1, decode = q4k._kernels()
    stage = q4k._stage_kernel()
    variants = [
        (w1, _w1_variants()),
        (decode, _r2_variants()),
        (stage, _stage_variants()),
    ]
    for fn, table in variants:
        for label, variant in table.items():
            _compile_variant(triton, fn, label, variant)
    print("q4 KV FRONTEND CHECK PASSED "
          "(W1 + R2 + R1-stage compile for sm_121 in every host "
          "dtype variant)")


if __name__ == "__main__":
    main()
