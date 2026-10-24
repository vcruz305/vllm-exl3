"""GB10 (sm_121) parity harness for the milestone-1 q4 KV cache.

Run ONLY on a DGX Spark GB10 with the MiniMax-M3.1-Flash vLLM image
(Triton + CUDA available). This VPS has no GPU and no triton: the GPU
cases are validated on GB10; the CPU parts of every case (input
construction, consistency validation, the pure-torch reference paths
and the gate functions themselves on synthetic perturbations) are
exercised on the VPS by ``--cpu-dry-run``.

Exact commands (from the plugin worktree root):

    python tools/gb10_q4kv_parity.py --json q4kv.json  # GB10 only
    python tools/gb10_q4kv_parity.py --cpu-dry-run     # any CPU (incl. VPS)

Structure (the GB10 val3 lesson: the harness must NEVER stop at the
first failure — one run must give the full picture). Every unit below
runs to completion, prints exactly one PASS/FAIL line, and records its
full diagnostics in a JSON receipt (``--json PATH``); the process exits
non-zero if any unit failed.

Units:
  0.  Frontend check (tools/q4kv_triton_frontend_check.py): W1 and R2
      must frontend-compile for sm_121 in every host dtype variant
      before any GPU math.
  1a. W1 norm+RoPE numerics: the kernel's post-norm+RoPE bf16 rows — the
      q_out buffer and the processed K / index_k rows it writes back into
      qkv in place — must stay within 2 bf16 ulp of the pure-torch
      reference everywhere, with cosine >= 0.99999. Bit-exactness is NOT
      the gate here: the kernel's hardware sqrt.approx and Triton's
      reduction order legitimately move a handful of fp32 ulps, which
      after bf16 rounding shows up as <= 1 ulp on a small fraction of
      elements (see tools/q4kv_gates.py for the bound analysis).
  1b. W1 packing byte-exactness, decoupled from the norm rounding: the
      reference packer run on the kernel's OWN processed rows (read back
      from qkv after the call; V is not normed) must reproduce the
      kernel's pool bytes — data nibbles AND e4m3 scale bytes —
      byte-for-byte. Both packers see bit-identical inputs there, so any
      differing byte is a real packing bug. Unmapped pool pages must
      stay zero.
  1c. W1 fp8 index outputs: the index_q bytes and the index_k scatter
      bytes are e4m3 encodings of fp32 in-register values (the vendor op
      does the same), so no bf16 readback exists to byte-compare against;
      they are gated on e4m3 code distance (<= 2 representable-value
      steps) and decoded cosine (>= 0.99999) against the reference.
  1d. W1 rewrite scope: only the K and index_k rows of qkv may change in
      place; the Q, V and index_q rows must be bit-identical to the
      input (nothing downstream consumes an in-place Q on the q4 path —
      the sparse layer reads its query from the dedicated q_out buffer).
  2.  R2 decode accuracy, with the VENDOR kernel as ground truth: on
      real-shaped random data, the original vendor decode kernel runs on
      (a) the real e4m3 fp8 pool (the production baseline) and (b) an
      UNQUANTIZED bf16 pool built from the same K/V (the ground truth),
      both invoked through the ONE mirrored-argument builder. Gates:
      cos(R2, vendor-bf16) >= 0.99 (the q4 quantization floor),
      cos(vendor-fp8, vendor-bf16) >= 0.999 (the invocation itself is
      sound -- the GB10 val7 prefill failure was exactly this class),
      cos(fp32 oracle, vendor-bf16) >= 0.999 (oracle drift fails
      loudly), no NaN/Inf.
  3.  Pool bytes: the allocated pool is 73728 B/page and 576
      B/token/layer for 4 KV heads (spec arithmetic).
  4.  CUDA-graph capture of a FULL decode step through the overlay's
      PUBLIC wrappers: the W1 wrapper writes the batch's new tokens into
      the q4 pool and the decode wrapper reads the pool (including those
      tokens) through R2, with the production unit-scale tensors on the
      device. Under the static-buffer rules every input is a preallocated
      CUDA tensor that is restored before each replay; the replay must
      reproduce the eager pass bit-identically in EVERY buffer the step
      writes AND must read the CURRENT bytes of both static input buffers
      (fresh new-token qkv rows and rotated history pages are copied in
      between replays). Any host sync in either wrapper (the boot20
      crash: the unit-scale .item()) aborts the capture itself and fails
      the unit with the CUDA error.
5.  Prefill staging (the boot18 crash path): a realistic
      multi-sequence chunk over W1-derived pool bytes (real partial-RoPE
      table, real e4m3 fp8 index cache, real per-(kv-head, query) top-k
      with sentinels). The overlay's prefill path on the q4 pool under a
      2 MiB page cap must be bit-identical to the same path under the
      default cap (the capped slice plan equals the uncapped staging).
      Ground truth is the ORIGINAL prefill kernel: it runs on (a) the
      real e4m3 fp8 pool built from the same K/V and (b) an UNQUANTIZED
      bf16 pool built from the same K/V, both through the ONE
      mirrored-argument builder. Gates: cos(q4 staged, vendor-bf16)
      >= 0.99, cos(fp8-original, vendor-bf16) >= 0.999, cos(fp32
      oracle, vendor-bf16) >= 0.999 (oracle drift fails loudly),
      finite outputs. --cpu-dry-run runs every reference part of the
      case (the capped==uncapped equality proof at the attention level,
      the slice-plan coverage checks, a meta-device audit of the
      staging primitives, and a structure audit of the ONE
      vendor-invocation builder -- the vendor legs themselves are
      GPU-only).

Every unit builds its inputs from tools/q4kv_harness_inputs.py and runs
that module's input-consistency validator BEFORE any kernel or reference
call. A harness data bug aborts the unit with a message naming the
input, never as an IndexError that looks like a kernel bug.

Device placement is explicit end to end (the GB10 val4 lesson: one CPU
output buffer reached the merge kernel and Triton died with an opaque
"Pointer argument cannot be accessed from Triton (cpu tensor?)"): the
one bundle mover ``_bundle_to_device`` moves each case's ENTIRE
kernel-facing bundle to the target device, the R2 output buffer is
allocated on the target, the pure-torch reference paths (fp8 baseline,
fp32 oracle) are CPU-only and refuse anything else, and the src
wrappers themselves fail closed naming any off-device argument. The
``--cpu-dry-run`` device audit simulates the move on the meta device to
prove the mover would leave nothing behind on the CPU.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import q4kv_gates as gates  # noqa: E402
import q4kv_harness_inputs as hi  # noqa: E402
from vllm_exl3 import minimax_q4kv_kernels as q4k  # noqa: E402
from vllm_exl3 import minimax_q4kv_reference as ref  # noqa: E402

# Real model shapes, shared with the CPU dry run (see q4kv_harness_inputs).
NQ, NKV, NIQ = hi.NQ, hi.NKV, hi.NIQ
ROTARY = hi.ROTARY
EPS = hi.EPS
BLOCK = hi.BLOCK
NUM_PAGES = hi.NUM_PAGES
NUM_REQS = hi.NUM_REQS
TOPK = hi.TOPK


def _bundle_to_device(bundle, device):
    """Move every tensor in a case-input bundle to ``device``.

    One mover for every case: it walks dicts/tuples/lists recursively and
    returns a NEW bundle with the same structure; non-tensor entries
    (scales, eps) pass through unchanged. The GPU cases move their whole
    kernel-facing bundle to CUDA with it, and --cpu-dry-run simulates
    the same move on the meta device to audit coverage, so nothing is
    ever left on the CPU because a builder forgot a key (the GB10 val4
    failure: one CPU output buffer reached the merge kernel and Triton
    died with an opaque pointer error).
    """
    if isinstance(bundle, torch.Tensor):
        return bundle.to(device)
    if isinstance(bundle, dict):
        return {k: _bundle_to_device(v, device) for k, v in bundle.items()}
    if isinstance(bundle, tuple):
        return tuple(_bundle_to_device(v, device) for v in bundle)
    if isinstance(bundle, list):
        return [_bundle_to_device(v, device) for v in bundle]
    return bundle


def _iter_tensors(bundle, prefix=""):
    """(path, tensor) for every tensor in a bundle, recursively."""
    if isinstance(bundle, torch.Tensor):
        yield prefix or "<root>", bundle
    elif isinstance(bundle, dict):
        for k, v in bundle.items():
            yield from _iter_tensors(
                v, f"{prefix}.{k}" if prefix else str(k)
            )
    elif isinstance(bundle, (tuple, list)):
        for i, v in enumerate(bundle):
            yield from _iter_tensors(v, f"{prefix}[{i}]")


# Exactly the tensors each Triton kernel (plus, for R2, the unchanged
# vLLM merge kernel) receives, by harness bundle key. Everything else
# in a case bundle is a reference-path input and stays on the CPU
# bundle, where the pure-torch oracles consume them.
_W1_KERNEL_KEYS = (
    "qkv", "norm_ws", "cos_sin", "positions", "slot_mapping",
    "index_slot_mapping", "pool", "index_cache", "q_out", "index_q_out",
)
_R2_KERNEL_KEYS = ("q", "pool", "topk", "block_table", "seq_lens")


def _kernel_bundle(inp: dict, keys, device) -> dict:
    """The kernel-facing slice of a VALIDATED case bundle, with every
    tensor moved to ``device`` by the one bundle mover."""
    return _bundle_to_device({k: inp[k] for k in keys}, device)


def _cos(a, b):
    return gates._cosine(a, b)


# ---------------------------------------------------------------------------
# Shared W1 evidence: run the Triton kernel and the reference ONCE, then let
# every 1x unit gate the same evidence. Construction failures propagate to
# each unit independently (each unit still gets its own verdict line).
# ---------------------------------------------------------------------------

_W1_EVIDENCE = None


def _w1_evidence():
    global _W1_EVIDENCE
    if _W1_EVIDENCE is None:
        _W1_EVIDENCE = _W1Evidence()
    return _W1_EVIDENCE


class _W1Evidence:
    """The kernel's outputs and the reference's outputs for the same inputs."""

    def __init__(self) -> None:
        inp = hi.w1_inputs()
        hi.validate_w1_inputs(inp)
        self.inp = inp
        n = inp["qkv"].shape[0]
        self.n = n
        # The ENTIRE kernel-facing bundle on one device, via the one
        # bundle mover (nothing may be moved tensor-by-tensor again:
        # a forgotten tensor is exactly the val4 failure class).
        gpu = _kernel_bundle(inp, _W1_KERNEL_KEYS, "cuda")
        nw = gpu["norm_ws"]
        q4k.q4_w1_insert(
            gpu["qkv"], nw["q"], nw["k"], gpu["cos_sin"], gpu["positions"],
            NQ, NKV, ROTARY, EPS, nw["iq"], nw["ik"], NIQ,
            gpu["slot_mapping"], gpu["index_slot_mapping"], gpu["pool"],
            gpu["index_cache"], BLOCK, gpu["q_out"], gpu["index_q_out"],
        )
        torch.cuda.synchronize()
        # The kernel's own outputs, read back after the call.
        self.qkv_after = gpu["qkv"].cpu()
        self.q_out = gpu["q_out"].cpu()
        self.index_q_bytes = (
            gpu["index_q_out"].cpu().view(torch.uint8)
        )
        self.pool_after = (
            gpu["pool"].cpu().reshape(NUM_PAGES, 2 * NKV, ref.SLOT_REGION)
        )
        self.index_cache_after = (
            gpu["index_cache"].cpu().view(torch.uint8)
        )
        # The reference paths (CPU, independent device path).
        nw = inp["norm_ws"]
        self.ref_out = ref.reference_w1_write(
            inp["qkv"], nw["q"], nw["k"], nw["iq"], nw["ik"],
            inp["cos_sin"], inp["positions"], EPS, ROTARY, NQ, NKV, NIQ,
            inp["slot_mapping"], inp["index_slot_mapping"],
        )
        x = inp["qkv"].to(torch.float32).view(n, -1, ref.HEAD_DIM)
        cs = (
            inp["cos_sin"].to(torch.float32)[inp["positions"]][:, :ROTARY]
        )
        self.qn = ref.rope_partial_neox(
            ref.gemma_rmsnorm(x[:, :NQ], nw["q"], EPS), cs
        )
        self.kn = ref.rope_partial_neox(
            ref.gemma_rmsnorm(x[:, NQ:NQ + NKV], nw["k"], EPS), cs
        )
        self.ikn = ref.rope_partial_neox(
            ref.gemma_rmsnorm(x[:, -1], nw["ik"], EPS), cs
        )

    def slots_after(self) -> torch.Tensor:
        return self.qkv_after.view(self.n, -1, ref.HEAD_DIM)

    def slots_before(self) -> torch.Tensor:
        return self.inp["qkv"].view(self.n, -1, ref.HEAD_DIM)


def unit_1a() -> dict:
    """Norm+RoPE numerics of q_out and the processed K/index_k rows."""
    ev = _w1_evidence()
    n = ev.n
    checks = [
        gates.gate_bf16_numerics(
            ev.q_out, ev.qn.to(torch.bfloat16), "q_out"
        ),
        gates.gate_bf16_numerics(
            ev.slots_after()[:, NQ:NQ + NKV].reshape(n * NKV, ref.HEAD_DIM),
            ev.kn.to(torch.bfloat16).reshape(n * NKV, ref.HEAD_DIM),
            "processed-K",
        ),
        gates.gate_bf16_numerics(
            ev.slots_after()[:, -1], ev.ikn.to(torch.bfloat16),
            "processed-index_k",
        ),
    ]
    return {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "summary": "; ".join(
            f"{c['label']}: max_ulp={c['max_ulp']} "
            f"mismatch={c['n_mismatch']}/{c['n_elements']} "
            f"cos={c['cosine']:.8f}"
            for c in checks
        ),
    }


def unit_1b() -> dict:
    """Packing byte-exactness on the kernel's OWN processed rows."""
    ev = _w1_evidence()
    n = ev.n
    n_pages = ev.ref_out["pool"].shape[0]
    # The kernel's own rows: the processed K it wrote back into qkv, and
    # the (un-normed) V rows it packed. bf16 rounding is idempotent, so
    # the reference packer sees exactly the fp32 values the kernel packed.
    k_rows = ev.slots_after()[:, NQ:NQ + NKV].to(torch.float32)
    v_rows = ev.slots_before()[
        :, NQ + NKV:NQ + 2 * NKV
    ].to(torch.float32)
    sm = ev.inp["slot_mapping"]
    want = torch.zeros_like(ev.ref_out["pool"])
    for head in range(NKV):
        want[:, 2 * head] = ref._scatter_q4_tokens(
            k_rows[:, head], sm, n_pages, 2 * head
        )
        want[:, 2 * head + 1] = ref._scatter_q4_tokens(
            v_rows[:, head], sm, n_pages, 2 * head + 1
        )
    checks = [
        gates.gate_bytes_exact(
            ev.pool_after[:n_pages], want,
            "pool data nibbles + e4m3 scale bytes",
        ),
    ]
    unmapped = ev.pool_after[n_pages:]
    unmapped_ok = bool(unmapped.numel() == 0 or int(unmapped.sum()) == 0)
    checks.append({
        "gate": "bytes-exact",
        "label": "unmapped pool pages stay zero",
        "passed": unmapped_ok,
        "reason": None if unmapped_ok else "kernel wrote unmapped pages",
        "n_elements": int(unmapped.numel()),
        "n_mismatch": 0 if unmapped_ok else 1,
    })
    failed = [c for c in checks if not c["passed"]]
    return {
        "passed": not failed,
        "checks": checks,
        "summary": (
            "reference packer on the kernel's own rows is byte-identical; "
            "unmapped pages zero"
            if not failed
            else "; ".join(f"{c['label']}: {c['reason']}" for c in failed)
        ),
    }


def unit_1c() -> dict:
    """fp8 index byte outputs: e4m3 code-distance gate (no bf16 readback
    exists for the fp32 values the vendor op and the reference encode)."""
    ev = _w1_evidence()
    ic_want = torch.zeros_like(ev.inp["index_cache"]).view(torch.uint8)
    valid = ev.inp["index_slot_mapping"] >= 0
    ism = ev.inp["index_slot_mapping"][valid]
    ic_want.view(-1, BLOCK)[ism] = ev.ref_out["index_k"][valid]
    checks = [
        gates.gate_e4m3_codes(
            ev.index_q_bytes, ev.ref_out["index_q"], "index_q bytes"
        ),
        gates.gate_e4m3_codes(
            ev.index_cache_after, ic_want, "index_k scatter bytes"
        ),
    ]
    return {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "summary": "; ".join(
            f"{c['label']}: max_code_distance={c['max_code_distance']} "
            f"mismatch={c['n_mismatch']}/{c['n_elements']} "
            f"cos={c['decoded_cosine']:.8f}"
            for c in checks
        ),
    }


def unit_1d() -> dict:
    """Rewrite scope: only the K and index_k rows of qkv may change."""
    ev = _w1_evidence()
    before = ev.slots_before()
    after = ev.slots_after()
    untouched = (
        list(range(0, NQ))
        + list(range(NQ + NKV, NQ + 2 * NKV))
        + list(range(NQ + 2 * NKV, NQ + 2 * NKV + NIQ))
    )
    changed = [
        i for i in untouched if not torch.equal(before[:, i], after[:, i])
    ]
    # K and index_k are rewritten in place (the readback gate 1a covers
    # their numerics); every other row must be bit-identical.
    k_changed = not torch.equal(
        before[:, NQ:NQ + NKV], after[:, NQ:NQ + NKV]
    )
    ik_changed = not torch.equal(before[:, -1], after[:, -1])
    passed = not changed and k_changed and ik_changed
    return {
        "passed": passed,
        "summary": (
            f"unexpectedly changed rows: {changed}"
            if changed
            else "only K/index_k rewritten in place"
        )
        + ("" if k_changed else "; processed K was NOT written back")
        + ("" if ik_changed else "; processed index_k was NOT written back"),
    }


# ---------------------------------------------------------------------------
# R2 (decode) units
# ---------------------------------------------------------------------------


def _fp8_baseline(fp8_pool, block_table, topk, q, sm_scale):
    """Exact attention over the fp8-quantized pool (baseline math).

    Reference path, CPU-only by construction: it consumes the same
    validated CPU bundle the oracles use, and a tensor that lives on the
    GPU side (or a simulated meta device) must never be handed in here.
    """
    for name, t in (
        ("fp8_pool", fp8_pool), ("block_table", block_table),
        ("topk", topk), ("q", q),
    ):
        if t.device.type != "cpu":
            raise ValueError(
                f"fp8 baseline is a CPU-only reference path: {name} is on "
                f"{t.device}"
            )
    total_q = q.shape[0]
    # The VENDOR pool layout keeps [K | V] in every token row
    # (sparse_attn.py:24-27): K = [..., :128], V = [..., 128:]. The val7
    # harness misread this pool as K/V-blocked head slots.
    values = fp8_pool.view(NUM_PAGES, NKV, BLOCK, 256).to(torch.float32)
    k_pages = values[..., :128].contiguous()
    v_pages = values[..., 128:].contiguous()
    # seq_lens = TOPK*BLOCK in R2, so the causal past ends at
    # TOPK*BLOCK - 1 for baseline and oracle alike.
    query_pos = torch.full((total_q,), TOPK * BLOCK - 1)
    flat_topk = topk[0]  # same selection for every KV head
    return ref.reference_sparse_attention(
        q.to(torch.float32), k_pages, v_pages, block_table.cpu(),
        flat_topk, query_pos, sm_scale,
    )


def _decode_fixture():
    inp = hi.decode_inputs()
    hi.validate_decode_inputs(inp)
    inp2 = hi.decode_inputs()
    # Deterministic builders: case 4 must see the very bytes case 2 ran
    # on (otherwise the eager/graph comparison is vacuous).
    for key, t in inp.items():
        if isinstance(t, torch.Tensor):
            assert torch.equal(t, inp2[key]), (
                f"decode fixture is not deterministic in {key}"
            )
    return inp


def _r2_static_bundle(inp: dict, device) -> dict:
    """Preallocated static buffers for R2 + the merge kernel.

    Every kernel-facing input is moved to ``device`` by the one bundle
    mover and the output buffer is ALLOCATED there (never inherited
    from the CPU bundle — the val4 failure). Case 4 (CUDA graph) copies
    new bytes into these static buffers before replay; case 2 runs the
    very same buffers eagerly.
    """
    gpu = _kernel_bundle(inp, _R2_KERNEL_KEYS, device)
    gpu["sm_scale"] = inp["sm_scale"]  # non-tensor passthrough
    gpu["output"] = torch.zeros_like(gpu["q"])
    return gpu


def _launch_r2(gpu: dict) -> None:
    """The only call site of the R2 wrapper (eager, warmup and capture)."""
    q4k.q4_sparse_decode(
        gpu["q"], gpu["pool"], gpu["topk"], gpu["block_table"],
        gpu["seq_lens"], NKV, gpu["sm_scale"], gpu["output"], 1,
    )


def _run_r2_bundle(gpu: dict) -> torch.Tensor:
    _launch_r2(gpu)
    torch.cuda.synchronize()
    return gpu["output"].cpu()


# ---------------------------------------------------------------------------
# Case 4: a FULL decode step through the overlay's PUBLIC wrappers (the
# boot20 crash path). The captured step is W1 (new-token write into the q4
# pool) followed by R2 (read of the same pool, including those tokens),
# exactly the layer sequence a captured vLLM decode step executes, with
# the production unit-scale tensors (pinned to 1.0 by
# set_default_quant_scales, nvidia/model.py:1078-79) on the device.
# ---------------------------------------------------------------------------

_W1_STEP_KEYS = (
    "qkv", "norm_ws", "positions", "cos_sin", "slot_mapping",
    "index_slot_mapping", "pool", "index_cache", "q_out", "index_q_out",
    "block_table", "topk", "seq_lens",
)
_STEP_WRITE_KEYS = (
    "output", "pool", "q_out", "index_q_out", "index_cache", "qkv",
)
# The output-like buffers zeroed before each run; the pool and qkv are
# restored INPUTS and are never zeroed.
_STEP_OUT_KEYS = ("output", "q_out", "index_q_out", "index_cache")

_W1_WRAPPER = None
_DECODE_WRAPPER = None


def _overlay_wrappers():
    """The overlay's PUBLIC wrappers, built exactly as install() builds
    them (``_make_hook_wrapper``), from the ORIGINAL callables imported
    BY VALUE here. Nothing is patched on any module, so the other units'
    originals stay pristine; the wrapper objects are the overlay's real
    routing code (pool detection, byte re-view, unit-scale gate, device
    gates, kernel dispatch). GPU-only (imports vllm)."""
    global _W1_WRAPPER, _DECODE_WRAPPER
    if _W1_WRAPPER is None:
        from vllm import _custom_ops as ops
        from vllm.models.minimax_m3.common.ops.sparse_attn import (
            minimax_m3_sparse_attn_decode as original_decode,
        )
        from vllm_exl3 import minimax_q4kv as overlay

        _W1_WRAPPER = overlay._make_hook_wrapper(
            "w1", ops.fused_minimax_m3_qknorm_rope_kv_insert
        )
        _DECODE_WRAPPER = overlay._make_hook_wrapper(
            "decode", original_decode
        )
        assert overlay._is_q4kv_wrapper(_W1_WRAPPER) and (
            _W1_WRAPPER.__q4kv_original__
            is ops.fused_minimax_m3_qknorm_rope_kv_insert
        ), "the W1 wrapper must preserve the original as delegation target"
        assert overlay._is_q4kv_wrapper(_DECODE_WRAPPER) and (
            _DECODE_WRAPPER.__q4kv_original__ is original_decode
        ), "the decode wrapper must preserve the original as delegation target"
    return _W1_WRAPPER, _DECODE_WRAPPER


def _decode_step_static_bundle(inp: dict, device) -> dict:
    """Preallocated static buffers for the full decode step: every input
    moved to ``device`` by the ONE bundle mover, the output buffer
    ALLOCATED there (never inherited from the CPU bundle), and the
    production unit-scale tensors built on the device so the captured
    step exercises the real scale path."""
    gpu = _kernel_bundle(inp, _W1_STEP_KEYS, device)
    gpu["sm_scale"] = inp["sm_scale"]  # non-tensor passthrough
    gpu["output"] = torch.zeros_like(gpu["q_out"])
    gpu["unit_scale"] = torch.ones((), device=device)
    return gpu


def _launch_decode_step(gpu: dict) -> None:
    """The ONLY call site of the overlay's public wrappers: one full
    decode step -- the W1 wrapper writes the batch's new tokens into the
    q4 pool, then the decode wrapper reads the pool (including those
    tokens) through R2. Any host sync in either wrapper aborts the CUDA
    graph capture with a CUDA error (the boot20 crash class), which fails
    this unit loudly instead of the serve."""
    w1, decode = _overlay_wrappers()
    nw = gpu["norm_ws"]
    # Positional order: the vendor op's signature (_W1_PARAMS).
    w1(
        gpu["qkv"], nw["q"], nw["k"], gpu["cos_sin"], gpu["positions"],
        NQ, NKV, ROTARY, EPS, nw["iq"], nw["ik"], NIQ,
        gpu["slot_mapping"], gpu["index_slot_mapping"], gpu["pool"],
        gpu["index_cache"], BLOCK, gpu["q_out"], gpu["index_q_out"],
        "nvfp4", False, None, None,
        gpu["unit_scale"], gpu["unit_scale"], True,
    )
    decode(
        gpu["q_out"], gpu["pool"], gpu["topk"], gpu["block_table"],
        gpu["seq_lens"], NKV, gpu["sm_scale"], gpu["output"], 1,
        k_scale=gpu["unit_scale"], v_scale=gpu["unit_scale"],
    )


def _step_snapshot(gpu: dict) -> dict:
    return {k: gpu[k].clone() for k in _STEP_WRITE_KEYS}


# ---------------------------------------------------------------------------
# ONE vendor-invocation builder (the GB10 val7 lesson)
# ---------------------------------------------------------------------------
# val7 measured cos(q4 staged prefill, fp8 ORIGINAL kernel) = 0.0022 while
# the staged path matched the fp32 oracle at 0.9952 and capped==uncapped
# was bit-identical: the HARNESS, not the overlay or the oracle, was
# wrong (verdict A). It had built the fp8 pool K/V-BLOCKED
# ([pages, 2*num_kv_heads, 128, 128]), while every ORIGINAL kernel reads
# the vendor layout [pages, num_kv_heads, 128, 2*head_dim] with [K | V]
# in every token row -- so the kernel's "V" pointer landed one token row
# over, inside the K data:
#   - sparse_attn.py:24-27 (the layout comment) and the K load at :161-169
#     vs the V load at :206-215 (offset head_dim + off_d);
#   - the writer csrc fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu:890
#     ("kv_cache logical shape [num_blocks, nkv, block_size,
#     2*head_dim]"; only the nvfp4 branch at :881-882 is slot-per-side);
#   - the allocator's per-token row is (head_size + head_size_v) wide,
#     v1/kv_cache_interface.py:557.
#
# Every invocation of a vendor kernel in this harness -- the fp8 baseline,
# the bf16 ground truth and the overlay wrapper (which shares the vendor
# signature) -- therefore goes through ONE argument builder that mirrors
# the REAL impl call site,
# vllm/models/minimax_m3/common/sparse_attention.py:447-521
# (MiniMaxM3SparseTritonImpl.forward):
#   :458-462  q viewed [-1, num_heads, head_dim];
#   :463-465  topk = topk_buffer[:num_tokens].transpose(0, 1), i.e. the
#             per-kv-head form [num_kv_heads, total_q, topk]; k/v scales
#             are passed ONLY for a quantized (fp8) cache, and the
#             production values are pinned to 1.0 by
#             set_default_quant_scales (nvidia/model.py:1078-79);
#   :486-499  the decode call (block_table, seq_lens, decode_query_len);
#   :502-521  the prefill call (cu_seqlens_q, seq_lens, context_lens as
#             prefix_lens, max_query_len).
# The builder refuses any pool outside the three geometries the wrappers
# accept -- vendor-layout fp8, vendor-layout bf16, or the q4 container
# for the overlay -- so the val7 layout bug can never recur, and it
# ties the pool to its invocation key so the q4 container can never
# reach the original kernel.

_VENDOR_POOL_KEYS = ("fp8_pool", "bf16_pool", "pool")
_VENDOR_ARG_KEYS = {
    # Every tensor slot a vendor (or overlay) wrapper receives, by
    # bundle key; the "pool" slot is filled from the pool_key argument
    # after the geometry gate.
    "prefill": ("q", "pool", "topk", "block_table", "cu_seqlens_q",
                "seq_lens", "prefix_lens"),
    "decode": ("q", "pool", "topk", "block_table", "seq_lens"),
}
_FP8_POOL_DTYPES = (
    torch.float8_e4m3fn, torch.float8_e4m3fnuz,
    torch.float8_e5m2, torch.float8_e5m2fnuz,
)


def _vendor_pool_gate(pool, pool_key: str) -> str:
    """Accept only a pool geometry the wrappers actually serve, tied to
    the pool_key that names which leg of the case it belongs to."""
    if pool_key not in _VENDOR_POOL_KEYS:
        raise ValueError(f"unknown vendor pool key {pool_key!r}")
    vendor_shape = tuple(pool.shape[1:]) == (NKV, BLOCK, 2 * 128)
    if pool.dtype in _FP8_POOL_DTYPES:
        if vendor_shape and pool_key == "fp8_pool":
            return "vendor-fp8"
    elif pool.dtype == torch.bfloat16:
        if vendor_shape and pool_key == "bf16_pool":
            return "vendor-bf16"
    elif pool.dtype == torch.uint8:
        if (
            pool_key == "pool"
            and tuple(pool.shape[1:]) == (2 * NKV, BLOCK, 72)
        ):
            return "q4-container"
    raise ValueError(
        "vendor invocation pool must be the vendor layout "
        f"[pages, {NKV}, {BLOCK}, 256] fp8/bf16 (pool_key 'fp8_pool' / "
        f"'bf16_pool') or the q4 container [pages, {2 * NKV}, {BLOCK}, "
        "72] uint8 (pool_key 'pool') -- the ORIGINAL kernels keep [K | "
        "V] in every token row (sparse_attn.py:24-27); "
        f"pool_key={pool_key!r} got {tuple(pool.shape)} {pool.dtype}"
    )


def _vendor_attention_args(kind: str, inp: dict, pool_key: str, device):
    """Build the COMPLETE vendor-invocation arguments (positional tuple,
    scale kwargs, output buffer) for the fp8 baseline, the bf16 ground
    truth and the overlay alike -- ONE builder, mirroring the real impl
    call site cited above. Returns (args, kwargs, out)."""
    if kind not in _VENDOR_ARG_KEYS:
        raise ValueError(f"unknown vendor call kind {kind!r}")
    pool_kind = _vendor_pool_gate(inp[pool_key], pool_key)
    src = {
        k: (inp[pool_key] if k == "pool" else inp[k])
        for k in _VENDOR_ARG_KEYS[kind]
    }
    moved = _bundle_to_device(src, device)
    assert isinstance(moved, dict)  # a dict in is a dict out

    def _t(key: str) -> torch.Tensor:
        v = moved[key]
        assert isinstance(v, torch.Tensor), f"{key} is not a tensor"
        return v

    out = torch.zeros_like(_t("q"))
    if kind == "prefill":
        # sparse_attention.py:502-521: (q, kv_cache, topk, block_table,
        # cu_seqlens_q, seq_lens, context_lens, max_query_len,
        # num_kv_heads, scale, out, k_scale, v_scale).
        args = (
            _t("q"), _t("pool"), _t("topk"), _t("block_table"),
            _t("cu_seqlens_q"), _t("seq_lens"), _t("prefix_lens"),
            inp["max_query_len"], NKV, inp["sm_scale"], out,
        )
    else:
        # sparse_attention.py:486-499: (q, kv_cache, topk, block_table,
        # seq_lens, num_kv_heads, scale, out, decode_query_len, k_scale,
        # v_scale). decode_query_len is 1 in this harness (one query
        # token per request, as the fixture pins).
        args = (
            _t("q"), _t("pool"), _t("topk"), _t("block_table"),
            _t("seq_lens"), NKV, inp["sm_scale"], out, 1,
        )
    if pool_kind == "vendor-fp8":
        # The real impl passes layer._k_scale/_v_scale for a quantized
        # cache; set_default_quant_scales pins them to 1.0, so the
        # production scale path is the unit scalar exercised here.
        kwargs = {
            "k_scale": torch.ones((), device=device),
            "v_scale": torch.ones((), device=device),
        }
    else:
        # :463-465: no scales for an unquantized (bf16) cache; the
        # overlay's wrappers additionally fail closed on non-unit scales.
        kwargs = {"k_scale": None, "v_scale": None}
    return args, kwargs, out


def _run_vendor(wrapper, kind: str, inp: dict, pool_key: str, device):
    """Invoke ``wrapper`` -- the ORIGINAL vendor kernel wrapper, or the
    overlay wrapper (which shares the vendor signature and routes a q4
    pool to the staged/R2 path) -- with arguments from the ONE builder,
    and return the raw output buffer on CPU."""
    args, kwargs, out = _vendor_attention_args(kind, inp, pool_key, device)
    wrapper(*args, **kwargs)
    torch.cuda.synchronize()
    return out.cpu()


def unit_2() -> dict:
    """R2 decode accuracy, with the VENDOR kernel as ground truth.

    The fp8 baseline is the ORIGINAL vendor decode kernel invoked through
    the ONE mirrored-argument builder (no pure-torch emulation), and the
    ground truth is the same kernel on an UNQUANTIZED bf16 pool built
    from the same K/V. Gates: cos(R2, vendor-bf16) >= 0.99 (the q4
    quantization floor -- a shared-assumption error in the fixture's
    conventions collapses this cosine), cos(vendor-fp8, vendor-bf16)
    >= 0.999 (the invocation itself is sound; the val7 prefill failure
    was exactly this class), cos(fp32 oracle, vendor-bf16) >= 0.999
    (oracle drift fails loudly), no NaN/Inf.
    """
    inp = _decode_fixture()
    gpu = _r2_static_bundle(inp, "cuda")
    out_r2 = _run_r2_bundle(gpu)
    del gpu
    from vllm.models.minimax_m3.common.ops.sparse_attn import (
        minimax_m3_sparse_attn_decode as original_decode,
    )

    out_truth = _run_vendor(original_decode, "decode", inp, "bf16_pool", "cuda")
    out_fp8 = _run_vendor(original_decode, "decode", inp, "fp8_pool", "cuda")
    oracle = ref.reference_sparse_attention(
        inp["q"].to(torch.float32),
        inp["k_vals"].contiguous(),
        inp["v_vals"].contiguous(),
        inp["block_table"], inp["topk"][0],
        torch.full((inp["q"].shape[0],), TOPK * BLOCK - 1),
        inp["sm_scale"],
    )
    cos_r2_truth = _cos(out_r2, out_truth)
    cos_fp8_truth = _cos(out_fp8, out_truth)
    cos_oracle_truth = _cos(oracle, out_truth)
    cos_r2_oracle = _cos(out_r2, oracle)
    finite = bool(
        torch.isfinite(out_r2).all()
        and torch.isfinite(out_truth).all()
        and torch.isfinite(out_fp8).all()
    )
    oracle_drift = cos_oracle_truth >= 0.999
    passed = (
        finite and oracle_drift
        and cos_r2_truth >= 0.99 and cos_fp8_truth >= 0.999
    )
    return {
        "passed": passed,
        "cos_r2_vs_vendor_bf16_truth": cos_r2_truth,
        "cos_vendor_fp8_vs_vendor_bf16_truth": cos_fp8_truth,
        "cos_oracle_vs_vendor_bf16_truth": cos_oracle_truth,
        "oracle_drift": oracle_drift,
        "cos_r2_vs_fp32_oracle": cos_r2_oracle,
        "output_finite": finite,
        "summary": (
            f"cos(R2, vendor-bf16)={cos_r2_truth:.6f}; "
            f"cos(vendor-fp8, vendor-bf16)={cos_fp8_truth:.6f}; "
            f"cos(oracle, vendor-bf16)={cos_oracle_truth:.6f}; "
            + ("ORACLE DRIFT: the fp32 oracle disagrees with the vendor "
               "kernel; " if not oracle_drift else "")
            + f"finite={finite}"
        ),
    }


def unit_3() -> dict:
    """Measured pool bytes must match the spec arithmetic."""
    page_bytes = NUM_PAGES * 2 * NKV * ref.SLOT_REGION
    per_token = ref.q4_pool_bytes_per_token(NKV)
    ok = (
        page_bytes == NUM_PAGES * ref.q4_pool_page_bytes(NKV)
        and per_token == 2 * NKV * 72 == 576
    )
    return {
        "passed": ok,
        "pool_bytes_per_page": ref.q4_pool_page_bytes(NKV),
        "pool_bytes_per_token_per_layer": per_token,
        "summary": (
            f"pool = {page_bytes} B total, {ref.q4_pool_page_bytes(NKV)} "
            f"B/page, {per_token} B/token/layer"
        ),
    }


def unit_4() -> dict:
    """A FULL decode step through the overlay's PUBLIC wrappers must be
    CUDA-graph capturable, and the replay must read the CURRENT bytes of
    every static input buffer.

    The captured step is the real layer sequence: the W1 wrapper writes
    the batch's new tokens (Gemma QK-norm + partial RoPE + q4 pack) into
    the q4 pool, then the decode wrapper reads the pool -- including
    those tokens -- through R2, with the production unit-scale tensors
    on the device. Static-buffer rules: every input is a preallocated
    CUDA tensor that is restored before each replay. The check is
    two-stage: (a) after the pristine bytes are restored, the replay
    must reproduce the eager pass bit-identically in EVERY buffer the
    step writes (output, pool, q_out, index bytes, qkv); (b) after NEW
    bytes are copied into BOTH static input buffers (fresh new-token qkv
    rows AND rotated history pages), the replay must reproduce the
    eager pass on THOSE bytes -- a stale replay, or a vacuous mutation,
    fails loudly. Any host sync in either wrapper (the boot20 crash:
    the unit-scale ``.item()``) aborts the capture itself and fails this
    unit with the CUDA error.
    """
    inp = hi.decode_step_inputs()
    hi.validate_decode_step_inputs(inp)
    gpu = _decode_step_static_bundle(inp, "cuda")
    pristine_qkv = gpu["qkv"].clone()
    pristine_pool = gpu["pool"].clone()

    def set_state(qkv_src, pool_src) -> None:
        gpu["qkv"].copy_(qkv_src)
        gpu["pool"].copy_(pool_src)
        for key in _STEP_OUT_KEYS:
            gpu[key].zero_()

    def run_eager() -> dict:
        # The eager pass IS vLLM's warm-up order: every captured forward
        # runs eagerly first (cudagraph_utils.py:464-467), so the unit
        # scales are validated and cached and the kernels compile here,
        # before anything is captured.
        _launch_decode_step(gpu)
        torch.cuda.synchronize()
        return _step_snapshot(gpu)

    eager_a = run_eager()
    finite = bool(torch.isfinite(eager_a["output"]).all())
    # Warmup on a side stream (torch requirement for capture).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _launch_decode_step(gpu)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _launch_decode_step(gpu)
    # Stage (a): pristine bytes restored, then the replay must reproduce
    # the eager pass bit-identically in every written buffer.
    set_state(pristine_qkv, pristine_pool)
    g.replay()
    torch.cuda.synchronize()
    replay_a = _step_snapshot(gpu)
    stage_a = all(
        bool(torch.equal(replay_a[k], eager_a[k])) for k in _STEP_WRITE_KEYS
    )
    # Stage (b): NEW bytes in BOTH static input buffers -- fresh
    # new-token qkv rows (the W1 leg's input) and rotated history pages
    # (the R2 leg's input). The eager pass on those bytes is the
    # reference; the replay after the same restore must reproduce it.
    b_qkv = torch.randn_like(gpu["qkv"]) * 0.5
    b_pool = torch.roll(pristine_pool, shifts=3, dims=0)
    set_state(b_qkv, b_pool)
    eager_b = run_eager()
    set_state(b_qkv, b_pool)
    g.replay()
    torch.cuda.synchronize()
    replay_b = _step_snapshot(gpu)
    stage_b = all(
        bool(torch.equal(replay_b[k], eager_b[k])) for k in _STEP_WRITE_KEYS
    )
    mutation_live = not bool(
        torch.equal(eager_b["output"], eager_a["output"])
    )
    passed = stage_a and stage_b and mutation_live and finite
    if passed:
        summary = (
            "the full decode step (W1 write + R2 read) captures cleanly; "
            "the replay reproduces the eager pass bit-identically in "
            "every written buffer and reads the current bytes of both "
            "static input buffers"
        )
    elif not finite:
        summary = "the eager decode step produced non-finite output"
    elif not stage_a:
        summary = "replay on the captured bytes differs from eager"
    elif not mutation_live:
        summary = "static-buffer mutation vacuous (outputs unchanged)"
    else:
        summary = "replay did not read the mutated static-buffer bytes"
    return {"passed": passed, "summary": summary}


# ---------------------------------------------------------------------------
# Case 5: prefill staging (the boot18 crash path)
# ---------------------------------------------------------------------------

def _prefill_reference_outputs(inp: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(fp8-baseline, fp32-oracle) prefill outputs, CPU-only reference
    (the --cpu-dry-run fixture-consistency check; the GB10 unit's fp8
    leg is the real vendor kernel, not this emulation)."""
    fp8_vals = inp["fp8_pool"].view(
        hi.PREFILL_NUM_PAGES, NKV, hi.BLOCK, 256
    ).to(torch.float32)
    # The VENDOR pool layout keeps [K | V] in every token row
    # (sparse_attn.py:24-27): K = [..., :128], V = [..., 128:]. The val7
    # harness misread this pool as K/V-blocked head slots.
    k8 = fp8_vals[..., :128].contiguous()
    v8 = fp8_vals[..., 128:].contiguous()
    base = ref.reference_prefill_attention(
        inp["q"].to(torch.float32), k8, v8, inp["block_table"],
        inp["topk"], inp["cu_seqlens_q"], inp["seq_lens"],
        inp["prefix_lens"], inp["sm_scale"],
    )
    oracle = ref.reference_prefill_attention(
        inp["q"].to(torch.float32), inp["k_vals"], inp["v_vals"],
        inp["block_table"], inp["topk"], inp["cu_seqlens_q"],
        inp["seq_lens"], inp["prefix_lens"], inp["sm_scale"],
    )
    return base, oracle


def unit_5() -> dict:
    """Prefill staging accuracy + memory bound (the boot18 crash path).

    A realistic multi-sequence chunk over W1-derived pool bytes. The
    overlay's prefill path on the q4 pool (page-capped slices under a
    2 MiB cap override, then the default cap) must (a) produce
    bit-identical outputs capped and uncapped. Ground truth is the
    ORIGINAL prefill kernel, invoked through the ONE mirrored-argument
    builder on (b) an UNQUANTIZED bf16 pool built from the same K/V and
    (c) the real e4m3 fp8 pool built from the same K/V. Gates:
    cos(q4 staged, vendor-bf16) >= 0.99, cos(fp8-original, vendor-bf16)
    >= 0.999, cos(fp32 oracle, vendor-bf16) >= 0.999 (oracle drift fails
    loudly), finite outputs.
    """
    inp = hi.prefill_inputs()
    hi.validate_prefill_inputs(inp)
    from vllm.models.minimax_m3.common.ops.sparse_attn import (
        minimax_m3_sparse_attn as original,
    )
    from vllm_exl3 import minimax_q4kv as q4ov

    overlay = q4ov._make_prefill_wrapper(original)
    # Ground truth: the ORIGINAL prefill kernel on the UNQUANTIZED bf16
    # pool built from the same K/V -- exactly the code path the overlay's
    # staging buffer feeds it, with no quantization noise at all.
    out_truth = _run_vendor(original, "prefill", inp, "bf16_pool", "cuda")
    # The production fp8 baseline: the ORIGINAL kernel on the real e4m3
    # pool (the val7 leg whose blocked pool produced cos 0.0022).
    out_fp8 = _run_vendor(original, "prefill", inp, "fp8_pool", "cuda")
    # (a) capped (2 MiB -> 8 pages) vs uncapped (default 48 MiB): the
    # sliced execution must equal the single-call staging bit-for-bit.
    import os

    os.environ["VLLM_EXL3_Q4KV_STAGING_MAX_MB"] = "2"
    out_capped = _run_vendor(overlay, "prefill", inp, "pool", "cuda")
    os.environ.pop("VLLM_EXL3_Q4KV_STAGING_MAX_MB", None)
    out_uncapped = _run_vendor(overlay, "prefill", inp, "pool", "cuda")
    bit_equal = bool(torch.equal(out_capped, out_uncapped))
    out = out_capped
    oracle = ref.reference_prefill_attention(
        inp["q"].to(torch.float32), inp["k_vals"].contiguous(),
        inp["v_vals"].contiguous(), inp["block_table"], inp["topk"],
        inp["cu_seqlens_q"], inp["seq_lens"], inp["prefix_lens"],
        inp["sm_scale"],
    )
    assert torch.isfinite(oracle).all()
    cos_q4_truth = _cos(out, out_truth)
    cos_fp8_truth = _cos(out_fp8, out_truth)
    cos_oracle_truth = _cos(oracle, out_truth)
    cos_q4_oracle = _cos(out, oracle)
    cos_q4_fp8 = _cos(out, out_fp8)
    finite = bool(torch.isfinite(out).all())
    oracle_drift = cos_oracle_truth >= 0.999
    passed = (
        bit_equal and finite and oracle_drift
        and cos_q4_truth >= 0.99 and cos_fp8_truth >= 0.999
    )
    return {
        "passed": passed,
        "capped_equals_uncapped": bit_equal,
        "cos_q4_staged_vs_vendor_bf16_truth": cos_q4_truth,
        "cos_fp8_original_vs_vendor_bf16_truth": cos_fp8_truth,
        "cos_oracle_vs_vendor_bf16_truth": cos_oracle_truth,
        "oracle_drift": oracle_drift,
        "cos_q4_staged_vs_fp32_oracle": cos_q4_oracle,
        "cos_q4_staged_vs_fp8_original": cos_q4_fp8,
        "output_finite": finite,
        "summary": (
            f"capped==uncapped bit-identical={bit_equal}; "
            f"cos(q4 staged, vendor-bf16)={cos_q4_truth:.6f}; "
            f"cos(fp8 original, vendor-bf16)={cos_fp8_truth:.6f}; "
            f"cos(oracle, vendor-bf16)={cos_oracle_truth:.6f}; "
            + ("ORACLE DRIFT: the fp32 oracle disagrees with the vendor "
               "kernel; " if not oracle_drift else "")
            + f"finite={finite}"
        ),
    }


def unit_6() -> dict:
    """Fast prefill staging (VLLM_EXL3_Q4KV_PREFILL_FAST=1) accuracy.

    The fast path must be OBSERVABLY IDENTICAL to the milestone-1 staged
    path: same slice plan, same staged bf16 bytes, therefore the same
    attention output bit for bit. Gates:
      - torch.equal(fast, legacy) over the full output buffer;
      - cos(fast, vendor-bf16 ground truth) >= 0.995 (tighter than case
        5's 0.99: this leg must not add any error at all);
      - cos(fp32 oracle, vendor-bf16) >= 0.999 (ORACLE DRIFT);
      - finite output.
    """
    inp = hi.prefill_inputs()
    hi.validate_prefill_inputs(inp)
    from vllm.models.minimax_m3.common.ops.sparse_attn import (
        minimax_m3_sparse_attn as original,
    )
    from vllm_exl3 import minimax_q4kv as q4ov

    overlay = q4ov._make_prefill_wrapper(original)
    out_truth = _run_vendor(original, "prefill", inp, "bf16_pool", "cuda")
    os.environ["VLLM_EXL3_Q4KV_PREFILL_FAST"] = "0"
    try:
        out_legacy = _run_vendor(overlay, "prefill", inp, "pool", "cuda")
    finally:
        os.environ.pop("VLLM_EXL3_Q4KV_PREFILL_FAST", None)
    os.environ["VLLM_EXL3_Q4KV_PREFILL_FAST"] = "1"
    try:
        out_fast = _run_vendor(overlay, "prefill", inp, "pool", "cuda")
    finally:
        os.environ.pop("VLLM_EXL3_Q4KV_PREFILL_FAST", None)
    oracle = ref.reference_prefill_attention(
        inp["q"].to(torch.float32), inp["k_vals"].contiguous(),
        inp["v_vals"].contiguous(), inp["block_table"], inp["topk"],
        inp["cu_seqlens_q"], inp["seq_lens"], inp["prefix_lens"],
        inp["sm_scale"],
    )
    bit_equal = bool(torch.equal(out_fast, out_legacy))
    finite = bool(torch.isfinite(out_fast).all())
    cos_fast_truth = _cos(out_fast, out_truth)
    cos_oracle_truth = _cos(oracle, out_truth)
    oracle_drift = cos_oracle_truth >= 0.999
    passed = (
        bit_equal and finite and oracle_drift and cos_fast_truth >= 0.995
    )
    return {
        "passed": passed,
        "fast_equals_legacy_bitwise": bit_equal,
        "cos_fast_vs_vendor_bf16_truth": cos_fast_truth,
        "cos_oracle_vs_vendor_bf16_truth": cos_oracle_truth,
        "oracle_drift": oracle_drift,
        "output_finite": finite,
        "summary": (
            f"fast==legacy bit-identical={bit_equal}; "
            f"cos(fast, vendor-bf16)={cos_fast_truth:.6f}; "
            f"cos(oracle, vendor-bf16)={cos_oracle_truth:.6f}; "
            + ("ORACLE DRIFT; " if not oracle_drift else "")
            + f"finite={finite}"
        ),
    }


# Prefix lengths of the measured serve (receipts/boot21: 6,532 / 12,071 /
# 24,798 prompt tokens -> 338 / 324 / 276 tok/s) rounded up to whole
# 128-token pages, plus the 32K production limit.
TIMING_PAGES = (64, 128, 240)  # 8K / 16K / 30K tokens
TIMING_REPS = 3
MODEL_LAYERS = 60  # boot21: layers 0..59
CHUNK_TOKENS = 1024  # max_num_batched_tokens


def _time_ms(fn, reps: int = TIMING_REPS) -> float:
    """Best-of-``reps`` wall time of ``fn`` in ms (sync around each)."""
    best = float("inf")
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def _staged_bytes_identical(inp: dict, cap: int, device) -> tuple[bool, str]:
    """Bit-compare the two staging paths on the exact bytes the vendor
    kernel reads: per slice, over ``buffer[:n+1]`` (the staged pages plus
    the trailing zero page), in the plan's page order.

    This is the claim ``VLLM_EXL3_Q4KV_PREFILL_FAST`` actually makes, and
    it is a claim about STAGED BYTES -- not about the attention output.
    The old gate compared the vendor kernel's output instead, which
    conflates three different things (slice plan, staged bytes, kernel
    determinism) and reports a bytes failure for any of them. Both plans
    are rebuilt here so a plan difference is also caught.
    """
    from vllm_exl3 import minimax_q4kv_kernels as q4k

    pool = inp["pool"].to(device)
    bt = inp["block_table"].to(device)
    topk = inp["topk"].to(device)
    cu = inp["cu_seqlens_q"].to(device)
    legacy = ref.plan_prefill_slices(pool, bt, topk, cu, NKV, cap)
    fast = ref.plan_prefill_slices_fast(pool, bt, topk, cu, NKV, cap)
    if len(legacy) != len(fast):
        return False, f"plan: {len(fast)} fast slices != {len(legacy)} legacy"
    shape = (cap + 1, NKV, BLOCK, 2 * 128)
    for i, (a, b) in enumerate(zip(legacy, fast)):
        if a["q"] != b["q"] or a["n"] != b["n"]:
            return False, f"slice {i}: q/n differ {a['q']}/{a['n']} vs {b['q']}/{b['n']}"
        if not torch.equal(a["pages"], b["pages"]):
            return False, f"slice {i}: page order differs"
        ba = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        bb = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        ref.fill_staging_buffer(ba, pool, a["pages"])
        q4k.q4_stage_pages(bb, pool, b["pages"])
        if not torch.equal(ba[: b["n"] + 1], bb[: b["n"] + 1]):
            bad = (ba[: b["n"] + 1] != bb[: b["n"] + 1]).nonzero()
            return False, (
                f"slice {i} (q={b['q']}, n={b['n']}): "
                f"{bad.shape[0]} of {ba[: b['n'] + 1].numel()} bf16 "
                f"elements differ, first at {tuple(bad[0].tolist())}"
            )
        del ba, bb
    return True, f"{len(fast)} slices, {sum(s['n'] for s in fast)} pages staged"


def unit_7() -> dict:
    """Per-chunk prefill time, legacy vs fast, at 8K / 16K / 30K prefixes.

    One layer of one 1024-token chunk per measurement (the harness has no
    engine); the receipt reports the per-layer time and the model-wide
    per-chunk figure using ``MODEL_LAYERS`` layers, plus the projected
    prefill tok/s. The reference leg is the ORIGINAL kernel on an
    UNQUANTIZED bf16 pool -- the no-staging cost, i.e. what the fp8 path
    pays (slightly conservative: bf16 K/V reads are 2x the fp8 bytes).

    Gates, in the order they matter: the staged bytes are bit-identical
    per slice at every length (hard -- that is the fast path's whole
    claim), fast is faster than legacy at every length (hard), and fast
    stays within 1.35x of the no-staging reference at the 30K row (the
    staged write is real work the unquantized path does not do; measured
    1.22x on GB10, so 1.35x leaves headroom for a noisy box without
    hiding a regression). The vendor output is compared too, but only
    against the kernel's own run-to-run determinism: if the vendor kernel
    is not bit-reproducible at these shapes a fast/legacy output
    difference says nothing about staging, so it is reported, not gated.
    """
    from vllm.models.minimax_m3.common.ops.sparse_attn import (
        minimax_m3_sparse_attn as original,
    )
    from vllm_exl3 import minimax_q4kv as q4ov

    overlay = q4ov._make_prefill_wrapper(original)
    rows = []
    problems = []
    for nb in TIMING_PAGES:
        inp = hi.long_prefill_inputs(nb, chunk=CHUNK_TOKENS)
        ref_ms = _time_ms(
            lambda: _run_vendor(original, "prefill", inp, "bf16_pool", "cuda")
        )

        def run_legacy():
            os.environ["VLLM_EXL3_Q4KV_PREFILL_FAST"] = "0"
            try:
                return _run_vendor(overlay, "prefill", inp, "pool", "cuda")
            finally:
                os.environ.pop("VLLM_EXL3_Q4KV_PREFILL_FAST", None)

        def run_fast():
            os.environ["VLLM_EXL3_Q4KV_PREFILL_FAST"] = "1"
            try:
                return _run_vendor(overlay, "prefill", inp, "pool", "cuda")
            finally:
                os.environ.pop("VLLM_EXL3_Q4KV_PREFILL_FAST", None)

        legacy_ms = _time_ms(run_legacy)
        fast_ms = _time_ms(run_fast)
        cap = ref.staging_max_pages(NKV)
        slices = ref.plan_prefill_slices_fast(
            inp["pool"].to("cuda"), inp["block_table"].to("cuda"),
            inp["topk"].to("cuda"), inp["cu_seqlens_q"].to("cuda"),
            NKV, cap,
        )
        same, bytes_detail = _staged_bytes_identical(inp, cap, "cuda")
        # Control: two independent legacy runs. If the vendor kernel is not
        # bit-reproducible at this shape, fast-vs-legacy output equality
        # says nothing about staging and must not gate.
        deterministic = bool(torch.equal(run_legacy(), run_legacy()))
        outputs_identical = bool(torch.equal(run_fast(), run_legacy()))
        row = {
            "prefix_tokens": nb * 128,
            "pages": nb,
            "slices": len(slices),
            "pages_staged": sum(s["n"] for s in slices),
            "staging_amplification": sum(s["n"] for s in slices) / nb,
            "layer_ms_no_staging": ref_ms,
            "layer_ms_legacy": legacy_ms,
            "layer_ms_fast": fast_ms,
            "chunk_s_legacy": legacy_ms * MODEL_LAYERS / 1e3,
            "chunk_s_fast": fast_ms * MODEL_LAYERS / 1e3,
            "tok_s_legacy": CHUNK_TOKENS / (legacy_ms * MODEL_LAYERS / 1e3),
            "tok_s_fast": CHUNK_TOKENS / (fast_ms * MODEL_LAYERS / 1e3),
            "tok_s_no_staging": CHUNK_TOKENS / (ref_ms * MODEL_LAYERS / 1e3),
            "staged_bytes_identical": same,
            "staged_bytes_detail": bytes_detail,
            "vendor_output_deterministic": deterministic,
            "vendor_outputs_identical": outputs_identical,
        }
        rows.append(row)
        if not same:
            problems.append(f"{nb} pages: staged bytes differ ({bytes_detail})")
        if fast_ms >= legacy_ms:
            problems.append(
                f"{nb} pages: fast {fast_ms:.3f} ms is not faster than "
                f"legacy {legacy_ms:.3f} ms"
            )
        if deterministic and not outputs_identical:
            problems.append(
                f"{nb} pages: staged bytes match but the vendor output "
                "differs (deterministic kernel, so this is a real "
                "invocation difference)"
            )
    top = rows[-1]
    # Staging overhead ceiling at the longest prefix: the fast path still
    # writes 328 KiB of staged bf16 per page that the unquantized
    # reference never writes. Measured 1.22x on GB10; 1.35x is the
    # regression bound, not a parity claim.
    if top["layer_ms_fast"] > 1.35 * top["layer_ms_no_staging"]:
        problems.append(
            f"30K row: fast {top['layer_ms_fast']:.3f} ms exceeds "
            f"1.35x the no-staging reference "
            f"{top['layer_ms_no_staging']:.3f} ms "
            f"({top['layer_ms_fast'] / top['layer_ms_no_staging']:.2f}x)"
        )
    return {
        "passed": not problems,
        "rows": rows,
        "problems": problems,
        "summary": (
            "; ".join(
                f"{r['prefix_tokens']}tok: {r['layer_ms_legacy']:.2f} -> "
                f"{r['layer_ms_fast']:.2f} ms/layer "
                f"({r['tok_s_legacy']:.0f} -> {r['tok_s_fast']:.0f} tok/s, "
                f"no-staging {r['tok_s_no_staging']:.0f}, "
                f"{r['slices']} slices/"
                f"{r['staging_amplification']:.2f}x staged"
                + (
                    ""
                    if r["vendor_output_deterministic"]
                    else ", vendor kernel NOT bit-reproducible"
                )
                + ")"
                for r in rows
            )
            if not problems
            else "; ".join(problems)
        ),
    }


# ---------------------------------------------------------------------------
# Runner: every unit runs to completion; one line per unit; JSON receipt;
# non-zero exit if anything failed.
# ---------------------------------------------------------------------------


def _short_error(exc: BaseException) -> str:
    lines = traceback.format_exc(limit=6).strip().splitlines()
    return lines[-1] if lines else f"{type(exc).__name__}: {exc}"


def run_units(units, json_path: str | None, header: str) -> int:
    results: list[dict] = []
    print(header)
    for name, fn in units:
        try:
            diag = fn()
            diag = dict(diag)
        except Exception as exc:  # noqa: BLE001 - one verdict per unit
            diag = {"passed": False, "error": _short_error(exc)}
        passed = bool(diag.get("passed"))
        entry = {"case": name, "status": "PASS" if passed else "FAIL"}
        entry.update(diag)
        results.append(entry)
        line = f"case {name}: {entry['status']}"
        if passed and diag.get("summary"):
            line += f" — {diag['summary']}"
        elif not passed:
            line += f" — {diag.get('reason') or diag.get('summary') or diag.get('error')}"
        print(line)
    verdict = "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL"
    receipt = {
        "harness": "gb10_q4kv_parity",
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "verdict": verdict,
        "results": results,
    }
    if json_path:
        Path(json_path).write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"receipt written to {json_path}")
    print(f"VERDICT: {verdict} ({sum(r['status'] == 'PASS' for r in results)}"
          f"/{len(results)} units passed)")
    return 0 if verdict == "PASS" else 1


# ---------------------------------------------------------------------------
# CPU dry run: build every case's inputs, validate them, run every
# pure-torch reference path on CPU (no triton, no GPU), and exercise the
# gate functions themselves on synthetic perturbations.
# ---------------------------------------------------------------------------


def _dry_w1_reference() -> dict:
    """The reference packer runs and writes exactly the mapped pages."""
    inp = hi.w1_inputs()
    hi.validate_w1_inputs(inp)
    nw = inp["norm_ws"]
    out = ref.reference_w1_write(
        inp["qkv"], nw["q"], nw["k"], nw["iq"], nw["ik"], inp["cos_sin"],
        inp["positions"], EPS, ROTARY, NQ, NKV, NIQ,
        inp["slot_mapping"], inp["index_slot_mapping"],
    )
    sm = inp["slot_mapping"][inp["slot_mapping"] >= 0]
    want_pages = sorted({int(p) for p in (sm // BLOCK).tolist()})
    n_ref = out["pool"].shape[0]
    assert n_ref == max(want_pages) + 1, "reference pool page count wrong"
    wrote = [
        p for p in range(n_ref) if int(out["pool"][p].sum()) > 0
    ]
    assert wrote == want_pages, (
        f"reference wrote pages {wrote}, expected {want_pages}"
    )
    # The written slot bytes must dequantize to exactly the quantized
    # processed rows (the scatter offsets land where the layout says).
    x = inp["qkv"].to(torch.float32).view(out["q"].shape[0], -1, 128)
    cos_sin = inp["cos_sin"].to(torch.float32)[inp["positions"]][
        :, :ROTARY
    ]
    kn = ref.rope_partial_neox(
        ref.gemma_rmsnorm(x[:, NQ:NQ + NKV], nw["k"], EPS), cos_sin
    )
    deq = ref.dequant_pool_pages(out["pool"])
    for i, sm_i in enumerate(inp["slot_mapping"].tolist()):
        if sm_i < 0:
            continue
        page, tok = sm_i // BLOCK, sm_i % BLOCK
        for head in range(NKV):
            got_k = deq[page, 2 * head, tok]
            want_k = ref.unpack_q4_row(*ref.pack_q4_row(kn[i, head]))
            assert torch.equal(got_k, want_k), f"K offsets t{i}h{head}"
            got_v = deq[page, 2 * head + 1, tok]
            want_v = ref.unpack_q4_row(
                *ref.pack_q4_row(x[i, NQ + NKV + head].to(torch.bfloat16))
            )
            assert torch.equal(got_v, want_v), f"V offsets t{i}h{head}"
    ikn = ref.rope_partial_neox(
        ref.gemma_rmsnorm(x[:, -1], nw["ik"], EPS), cos_sin
    )
    assert torch.equal(out["index_k"], ref._e4m3_clamped(ikn))
    return {
        "passed": True,
        "summary": "inputs valid; reference packer runs and writes "
        "exactly the mapped pages/slots",
    }


def _dry_gate_a() -> dict:
    """Gate (a) self-test on the reference's own rows: a sparse 1-ulp
    perturbation must PASS, gross corruption must FAIL."""
    inp = hi.w1_inputs()
    hi.validate_w1_inputs(inp)
    nw = inp["norm_ws"]
    n = inp["qkv"].shape[0]
    x = inp["qkv"].to(torch.float32).view(n, -1, ref.HEAD_DIM)
    cs = inp["cos_sin"].to(torch.float32)[inp["positions"]][:, :ROTARY]
    base = ref.rope_partial_neox(
        ref.gemma_rmsnorm(x[:, :NQ], nw["q"], EPS), cs
    ).to(torch.bfloat16)
    scenarios = []
    # 1) sparse 1-ulp perturbation (the expected kernel-vs-reference
    #    noise shape): must pass.
    sparse = gates.bf16_ulp_step(base, +1, fraction=1 / 1024, seed=0)
    scenarios.append(
        ("sparse-1ulp-must-pass",
         gates.gate_bf16_numerics(sparse, base, "selftest"), True)
    )
    # 2) identical rows: must pass.
    scenarios.append(
        ("identical-must-pass",
         gates.gate_bf16_numerics(base, base, "selftest"), True)
    )
    # 3) gross corruption (1% additive): must fail.
    gross = base + (torch.randn(base.shape) * 0.05).to(torch.bfloat16)
    scenarios.append(
        ("gross-must-fail",
         gates.gate_bf16_numerics(gross, base, "selftest"), False)
    )
    # 4) one-sided NaN: must fail.
    nan = base.clone()
    nan.view(-1)[0] = float("nan")
    scenarios.append(
        ("one-sided-nan-must-fail",
         gates.gate_bf16_numerics(nan, base, "selftest"), False)
    )
    bad = [
        (name, c["passed"])
        for name, c, want in scenarios
        if c["passed"] != want
    ]
    return {
        "passed": not bad,
        "summary": (
            "gate behaves as specified on all 4 scenarios"
            if not bad
            else f"gate misbehaved on {bad}"
        ),
        "scenarios": [
            {"scenario": name, "expected_pass": want, **c}
            for name, c, want in scenarios
        ],
    }


def _dry_gate_b() -> dict:
    """Gate (b) self-test: byte-identical must pass; a single flipped
    nibble and a single flipped scale byte must fail."""
    inp = hi.w1_inputs()
    hi.validate_w1_inputs(inp)
    nw = inp["norm_ws"]
    out = ref.reference_w1_write(
        inp["qkv"], nw["q"], nw["k"], nw["iq"], nw["ik"], inp["cos_sin"],
        inp["positions"], EPS, ROTARY, NQ, NKV, NIQ,
        inp["slot_mapping"], inp["index_slot_mapping"],
    )
    pool = out["pool"]
    scenarios = [
        ("identical-must-pass",
         gates.gate_bytes_exact(pool, pool, "selftest"), True),
    ]
    # A single flipped data nibble (offset 0 of the packed region).
    flipped = pool.clone()
    flipped.view(-1)[0] ^= 0x01
    scenarios.append(
        ("single-flipped-nibble-must-fail",
         gates.gate_bytes_exact(flipped, pool, "selftest"), False),
    )
    # A single flipped e4m3 scale byte (inside the scale region).
    flipped = pool.clone()
    flipped.view(-1)[ref.DATA_REGION + 17] ^= 0x01
    scenarios.append(
        ("single-flipped-scale-byte-must-fail",
         gates.gate_bytes_exact(flipped, pool, "selftest"), False),
    )
    bad = [
        (name, c["passed"])
        for name, c, want in scenarios
        if c["passed"] != want
    ]
    return {
        "passed": not bad,
        "summary": (
            "gate behaves as specified on all 3 scenarios"
            if not bad
            else f"gate misbehaved on {bad}"
        ),
        "scenarios": [
            {"scenario": name, "expected_pass": want, **c}
            for name, c, want in scenarios
        ],
    }


def _dry_gate_c() -> dict:
    """Gate (c) self-test: byte-identical and sparse 1-code-step noise
    must pass; a 3-code-step jump and a one-sided NaN code must fail."""
    inp = hi.w1_inputs()
    hi.validate_w1_inputs(inp)
    base = torch.zeros((64, ref.HEAD_DIM), dtype=torch.uint8)
    base.view(-1).random_(0, 0x7F)  # valid finite e4m3 magnitude codes
    scenarios = [
        ("identical-must-pass",
         gates.gate_e4m3_codes(base, base, "selftest"), True),
    ]
    sparse = gates.e4m3_code_step(
        base, +1, fraction=4 / base.numel(), seed=0
    )
    scenarios.append(
        ("sparse-1-code-step-must-pass",
         gates.gate_e4m3_codes(sparse, base, "selftest"), True),
    )
    jump = gates.e4m3_code_step(
        base, +3, fraction=4 / base.numel(), seed=1
    )
    scenarios.append(
        ("3-code-step-jump-must-fail",
         gates.gate_e4m3_codes(jump, base, "selftest"), False),
    )
    nan = base.clone()
    nan.view(-1)[0] = 0x7F  # the NaN code on one side only
    scenarios.append(
        ("one-sided-nan-code-must-fail",
         gates.gate_e4m3_codes(nan, base, "selftest"), False),
    )
    bad = [
        (name, c["passed"])
        for name, c, want in scenarios
        if c["passed"] != want
    ]
    return {
        "passed": not bad,
        "summary": (
            "gate behaves as specified on all 4 scenarios"
            if not bad
            else f"gate misbehaved on {bad}"
        ),
        "scenarios": [
            {"scenario": name, "expected_pass": want, **c}
            for name, c, want in scenarios
        ],
    }


def _dry_decode() -> dict:
    """Decode fixture validity: the fp8 baseline and the fp32 oracle run
    to completion on CPU and agree before any kernel runs."""
    inp = hi.decode_inputs()
    hi.validate_decode_inputs(inp)
    base = _fp8_baseline(
        inp["fp8_pool"], inp["block_table"], inp["topk"], inp["q"],
        inp["sm_scale"],
    )
    oracle = ref.reference_sparse_attention(
        inp["q"].to(torch.float32), inp["k_vals"].contiguous(),
        inp["v_vals"].contiguous(), inp["block_table"], inp["topk"][0],
        torch.full((NUM_REQS,), TOPK * BLOCK - 1), inp["sm_scale"],
    )
    assert base.shape == oracle.shape == (NUM_REQS, NQ, 128)
    assert torch.isfinite(base).all(), "fp8 baseline produced NaN/inf"
    assert torch.isfinite(oracle).all(), "fp32 oracle produced NaN/inf"
    cos = _cos(base, oracle)
    return {
        "passed": cos >= 0.99,
        "cos_baseline_oracle": cos,
        "summary": f"cos(fp8 baseline, fp32 oracle) = {cos:.6f}; "
        "both reference paths run to completion",
    }


def _dry_device_audit() -> dict:
    """Simulated GPU-boundary audit (no GPU on this VPS): for the GPU
    path, the one bundle mover must put EVERY tensor of each case's
    kernel-facing bundle on the target device (simulated with the meta
    device), keep shapes/dtypes/counts intact, and pass non-tensors
    through unchanged; the output buffer must be allocated on the
    target; and the src wrappers must name every off-device argument
    instead of letting Triton fail opaquely."""
    device = torch.device("meta")
    problems: list[str] = []

    # (a) mover coverage: nothing in a kernel-facing bundle may stay on
    #     the CPU (or silently change) when the bundle is moved.
    for name, build, validate, keys in (
        ("w1", hi.w1_inputs, hi.validate_w1_inputs, _W1_KERNEL_KEYS),
        ("r2", hi.decode_inputs, hi.validate_decode_inputs,
         _R2_KERNEL_KEYS),
    ):
        inp = build()
        validate(inp)
        before = dict(_iter_tensors({k: inp[k] for k in keys}))
        moved = _kernel_bundle(inp, keys, device)
        after = dict(_iter_tensors(moved))
        if set(before) != set(after):
            problems.append(f"{name}: mover dropped/duplicated tensors")
        for path, b in before.items():
            a = after.get(path)
            if a is None:
                continue
            if a.device != device:
                problems.append(f"{name}/{path} stayed on {a.device}")
            if a.shape != b.shape or a.dtype != b.dtype:
                problems.append(f"{name}/{path} changed shape/dtype")

    # (b) the R2 static bundle: the OUTPUT buffer is allocated on the
    #     target (the val4 bug was a CPU output buffer), and the
    #     non-tensor sm_scale passes through unchanged.
    dec = hi.decode_inputs()
    hi.validate_decode_inputs(dec)
    static = _r2_static_bundle(dec, device)
    if static["output"].device != device or (
        static["output"].shape != dec["q"].shape
    ):
        problems.append("r2: output buffer not allocated on the target")
    if static["sm_scale"] != dec["sm_scale"]:
        problems.append("r2: sm_scale passthrough broken")
    for key in _R2_KERNEL_KEYS:
        if static[key].device != device:
            problems.append(f"r2/{key} not on the target device")

    # (c) non-tensor entries pass through the mover unchanged.
    passthrough = _bundle_to_device(
        {"sm_scale": 0.08, "eps": 1e-4, "none": None}, device
    )
    if passthrough != {"sm_scale": 0.08, "eps": 1e-4, "none": None}:
        problems.append("mower mutated non-tensor entries")

    # (d) the src device gate names every off-device argument: a CUDA
    #     reference device against CPU tensors must flag each one.
    cuda_ref = torch.device("cuda:0")
    r2_named = (
        ("pool", dec["pool"]), ("topk_idx", dec["topk"]),
        ("block_table", dec["block_table"]), ("seq_lens", dec["seq_lens"]),
        ("output", torch.zeros_like(dec["q"])),
    )
    flagged = [n for n, _ in q4k._off_device(cuda_ref, r2_named)]
    if flagged != [n for n, _ in r2_named]:
        problems.append(f"r2 gate did not name every argument: {flagged}")
    #     and only the misplaced one when the rest are on the reference.
    #     (a meta reference device stands in for cuda here: every CPU
    #     tensor is off-device relative to it, exactly as on GB10.)
    mixed = [
        (n, t.to(device) if n != "output" else t)
        for n, t in r2_named
    ]
    flagged = [n for n, _ in q4k._off_device(device, mixed)]
    if flagged != ["output"]:
        problems.append(f"r2 gate misreported the mixed bundle: {flagged}")

    # (e) end to end: the real R2 wrapper with an all-CPU bundle fails
    #     closed BEFORE any triton import, naming the query argument.
    try:
        q4k.q4_sparse_decode(
            dec["q"], dec["pool"], dec["topk"], dec["block_table"],
            dec["seq_lens"], NKV, dec["sm_scale"],
            torch.zeros_like(dec["q"]), 1,
        )
        problems.append("r2 wrapper accepted an all-CPU bundle")
    except ValueError as exc:
        msg = str(exc)
        if "'q'" not in msg or "cuda" not in msg:
            problems.append(f"r2 refusal does not name the argument: {msg}")
    #     and the W1 wrapper does the same for its query tensor.
    w1 = hi.w1_inputs()
    hi.validate_w1_inputs(w1)
    try:
        nw = w1["norm_ws"]
        q4k.q4_w1_insert(
            w1["qkv"], nw["q"], nw["k"], w1["cos_sin"], w1["positions"],
            hi.NQ, hi.NKV, hi.ROTARY, hi.EPS, nw["iq"], nw["ik"], hi.NIQ,
            w1["slot_mapping"], w1["index_slot_mapping"], w1["pool"],
            w1["index_cache"], hi.BLOCK, w1["q_out"], w1["index_q_out"],
        )
        problems.append("w1 wrapper accepted an all-CPU bundle")
    except ValueError as exc:
        msg = str(exc)
        if "'qkv'" not in msg or "cuda" not in msg:
            problems.append(f"w1 refusal does not name the argument: {msg}")

    # (f) the fp8 baseline is a CPU-only reference path: a target-device
    #     tensor handed to it must fail closed with a named error.
    try:
        _fp8_baseline(
            dec["fp8_pool"].to(device), dec["block_table"], dec["topk"],
            dec["q"], dec["sm_scale"],
        )
        problems.append("fp8 baseline accepted a target-device tensor")
    except ValueError as exc:
        if "fp8_pool" not in str(exc):
            problems.append(f"fp8 baseline refusal unnamed: {exc}")

    return {
        "passed": not problems,
        "problems": problems,
        "summary": (
            "bundle mover covers every kernel-facing tensor; output "
            "buffers are allocated on the target; the src device gate "
            "names every off-device argument"
            if not problems
            else "; ".join(problems)
        ),
    }


def _dry_prefill() -> dict:
    """Case 5's reference parts on CPU (no GPU, no triton, no vllm):
    the fixture is valid, the fp8-baseline and fp32-oracle prefill
    reference paths run to completion, the page-capped sliced execution
    equals the uncapped staging bit-for-bit, the plan partitions the
    chunk under the cap without dropping anything, and every
    serve-reachable tensor follows the pool's device (meta-device audit
    of the prefill bundle and the staging primitives)."""
    inp = hi.prefill_inputs()
    hi.validate_prefill_inputs(inp)
    problems: list[str] = []
    base, oracle = _prefill_reference_outputs(inp)
    if not (torch.isfinite(base).all() and torch.isfinite(oracle).all()):
        problems.append("fp8 baseline or fp32 oracle produced NaN/inf")
    cos_fixture = _cos(base, oracle)
    if cos_fixture < 0.99:
        problems.append(
            f"fixture sanity: cos(fp8 baseline, fp32 oracle)={cos_fixture:.6f}"
        )
    # (a) capped == uncapped, at the attention level, on this fixture.
    staging, fake_bt = ref.build_prefill_staging(
        inp["pool"], inp["block_table"], inp["topk"], NKV
    )
    out_uncapped = ref.reference_prefill_attention(
        inp["q"].to(torch.float32),
        staging[:, :, :, :128].to(torch.float32),
        staging[:, :, :, 128:].to(torch.float32),
        fake_bt, inp["topk"], inp["cu_seqlens_q"], inp["seq_lens"],
        inp["prefix_lens"], inp["sm_scale"],
    )
    cap = ref.staging_max_pages(NKV, max_mb=2.0)
    slices = ref.plan_prefill_slices(
        inp["pool"], inp["block_table"], inp["topk"],
        inp["cu_seqlens_q"], NKV, cap,
    )
    if len(slices) < 2:
        problems.append("the 2 MiB cap did not force slicing")
    pos = 0
    cu_list = inp["cu_seqlens_q"].tolist()
    prefix_list = inp["prefix_lens"].tolist()
    seq_list = inp["seq_lens"].tolist()
    out_capped = torch.zeros_like(out_uncapped)
    for s in slices:
        qs, qe = s["q"]
        if qs != pos or qe <= qs:
            problems.append(f"slice query range [{qs}, {qe}) is not contiguous")
        pos = qe
        if s["n"] > cap and qe - qs > 1:
            problems.append(f"slice of {s['n']} pages exceeds the {cap}-page cap")
        buffer = torch.zeros(
            (s["n"] + 1, NKV, 128, 256), dtype=torch.bfloat16
        )
        n = ref.fill_staging_buffer(buffer, inp["pool"], s["pages"])
        fake = ref.slice_fake_block_table(
            inp["block_table"], s["reqs"], s["pages"], n,
            inp["pool"].shape[0],
        )
        cu_new, prefix_new, _ = ref.prefill_slice_metadata(
            s, cu_list, prefix_list, torch.device("cpu")
        )
        part = ref.reference_prefill_attention(
            inp["q"][qs:qe].to(torch.float32),
            buffer[:n + 1, :, :, :128].to(torch.float32),
            buffer[:n + 1, :, :, 128:].to(torch.float32),
            fake, inp["topk"][:, qs:qe, :], cu_new,
            torch.tensor(
                seq_list[s["reqs"][0]:s["reqs"][-1] + 1], dtype=torch.int32
            ),
            prefix_new, inp["sm_scale"],
        )
        out_capped[qs:qe] = part
    if pos != inp["q"].shape[0]:
        problems.append("slices do not partition the chunk query axis")
    if not torch.equal(out_capped, out_uncapped):
        problems.append("capped sliced execution differs from uncapped")
    cos_q4 = _cos(out_uncapped, oracle)
    if cos_q4 < 0.99:
        problems.append(f"cos(q4 staged prefill, fp32 oracle)={cos_q4:.6f}")
    # (b) meta-device audit of the staging primitives: nothing
    # serve-reachable may stay on (or silently move back to) the CPU
    # when the pool is on the accelerator. (The vendor-invocation
    # builder's own structure audit is dry/6.)
    device = torch.device("meta")
    pool_m = inp["pool"].to(device)
    try:
        for side in (0, 1):
            t = ref._scale_offset_table(side, ref.POOL_BLOCK_SIZE, device=device)
            if t.device != device:
                problems.append(f"scale-offset table {side} on {t.device}")
        v = ref.dequant_pool_pages(
            pool_m.reshape(
                pool_m.shape[0], pool_m.shape[1], ref.SLOT_REGION
            )
        )
        if v.device != device:
            problems.append(f"dequant output on {v.device}")
        buf_m = torch.zeros(
            (3, NKV, 128, 256), dtype=torch.bfloat16, device=device
        )
        pages_m = torch.zeros((2,), dtype=torch.int64, device=device)
        n = ref.fill_staging_buffer(buf_m, pool_m, pages_m)
        if n != 2 or not buf_m[2].shape == (NKV, 128, 256):
            problems.append("fill_staging_buffer misbehaved on meta")
        fake_m = ref.slice_fake_block_table(
            inp["block_table"].to(device), [0], pages_m, n,
            inp["pool"].shape[0],
        )
        if fake_m.device != device:
            problems.append(f"slice fake table on {fake_m.device}")
        bm, phys, roq = ref._selected_page_bitmap(
            pool_m, inp["block_table"].to(device), inp["topk"].to(device),
            inp["cu_seqlens_q"].to(device),
        )
        for name, t in (
            ("request bitmap", bm), ("phys", phys), ("req_of_q", roq)
        ):
            if t.device != device:
                problems.append(f"bitmap/{name} on {t.device}")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"meta staging audit raised: {exc}")
    # the overlay prefill wrapper refuses an all-CPU bundle BEFORE any
    # vllm import, naming the pool argument.
    from vllm_exl3 import minimax_q4kv as q4ov

    calls: list = []
    wrapper = q4ov._make_prefill_wrapper(lambda *a, **k: calls.append(a))
    try:
        wrapper(
            inp["q"], inp["pool"], inp["topk"], inp["block_table"],
            inp["cu_seqlens_q"], inp["seq_lens"], inp["prefix_lens"],
            inp["max_query_len"], NKV, inp["sm_scale"],
            torch.zeros_like(inp["q"]),
        )
        problems.append("prefill wrapper accepted an all-CPU bundle")
    except ValueError as exc:
        msg = str(exc)
        if "'kv_cache'" not in msg or "cuda" not in msg:
            problems.append(f"prefill refusal does not name the pool: {msg}")
    # the prefill oracle is CPU-only: a meta tensor must fail closed.
    try:
        ref.reference_prefill_attention(
            inp["q"].to(device), inp["k_vals"], inp["v_vals"],
            inp["block_table"], inp["topk"], inp["cu_seqlens_q"],
            inp["seq_lens"], inp["prefix_lens"], inp["sm_scale"],
        )
        problems.append("prefill oracle accepted a meta tensor")
    except ValueError as exc:
        if "q" not in str(exc):
            problems.append(f"prefill oracle refusal unnamed: {exc}")
    return {
        "passed": not problems,
        "problems": problems,
        "cos_fp8_baseline_vs_oracle": cos_fixture,
        "cos_q4_staged_vs_oracle": cos_q4,
        "slices_under_2MiB_cap": len(slices),
        "summary": (
            f"fixture valid; capped==uncapped bit-identical; "
            f"cos(fp8, oracle)={cos_fixture:.6f}; "
            f"cos(q4 staged, oracle)={cos_q4:.6f}; "
            f"{len(slices)} slices under the 2 MiB cap; the meta device "
            "audit found no CPU-resident serve tensor"
            if not problems
            else "; ".join(problems)
        ),
    }


def _dry_vendor_args_audit() -> dict:
    """Structure audit of the ONE vendor-invocation builder (the vendor
    kernels themselves are GPU-only; this is the --cpu-dry-run share of
    the val7 follow-up). On the meta device the builder must move EVERY
    argument, preserve shapes/dtypes, allocate the output buffer on the
    target, pass the pinned unit scales only for the vendor fp8 pool and
    none for the bf16/q4 pools, put the arguments in exactly the real
    impl's positional order, and refuse the val7 K/V-blocked pool and
    every pool/key mismatch."""
    device = torch.device("meta")
    problems: list[str] = []

    def audit(kind: str, build, validate) -> None:
        inp = build()
        validate(inp)
        # The positional order the real impl passes
        # (sparse_attention.py:486-499 decode, :502-521 prefill).
        if kind == "prefill":
            order = ("q", "pool", "topk", "block_table", "cu_seqlens_q",
                     "seq_lens", "prefix_lens")
            arity = 11
            out_pos = 10
        else:
            order = ("q", "pool", "topk", "block_table", "seq_lens")
            arity = 9
            out_pos = 7
        originals = {k: inp[k] for k in order}
        for pool_key in ("fp8_pool", "bf16_pool", "pool"):
            args, kwargs, out = _vendor_attention_args(
                kind, inp, pool_key, device
            )
            if len(args) != arity:
                problems.append(f"{kind}: wrong positional arity {len(args)}")
                continue
            # the pool slot must carry exactly the pool this leg named
            originals["pool"] = inp[pool_key]
            for pos, key in enumerate(order):
                got = args[pos]
                want = originals[key]
                if not isinstance(got, torch.Tensor):
                    problems.append(f"{kind}/{key} is not a tensor")
                    continue
                if got.device != device:
                    problems.append(
                        f"{kind}/{key} ({pool_key}) stayed on {got.device}"
                    )
                if got.shape != want.shape or got.dtype != want.dtype:
                    problems.append(
                        f"{kind}/{key} ({pool_key}) changed shape/dtype"
                    )
            if (
                out.device != device
                or out.shape != inp["q"].shape
                or out.dtype != inp["q"].dtype
            ):
                problems.append(f"{kind}: output buffer wrong on {device}")
            if args[out_pos] is not out:
                problems.append(f"{kind}: output is not the last tensor arg")
            scales_fp8 = pool_key == "fp8_pool"
            for name in ("k_scale", "v_scale"):
                scale = kwargs[name]
                if scales_fp8:
                    if (
                        not isinstance(scale, torch.Tensor)
                        or scale.device != device
                        or tuple(scale.shape) != ()
                    ):
                        problems.append(
                            f"{kind}/{pool_key}: {name} is not a scalar on "
                            "the target"
                        )
                elif scale is not None:
                    problems.append(
                        f"{kind}/{pool_key}: {name} must be None for an "
                        "unquantized pool"
                    )
        # The pinned unit scale VALUE (meta tensors carry no data to
        # read, so the value check runs on a cpu build of the args).
        _, kw_cpu, _ = _vendor_attention_args(kind, inp, "fp8_pool", "cpu")
        for name in ("k_scale", "v_scale"):
            scale = kw_cpu[name]
            if (
                not isinstance(scale, torch.Tensor)
                or scale.numel() != 1
                or scale.item() != 1.0
            ):
                problems.append(
                    f"{kind}: fp8 {name} is not the pinned unit scalar"
                )
        # The val7 layout bug must fail closed: a K/V-blocked pool is
        # refused by the builder AND by the fixture validator.
        blocked = inp["fp8_pool"].reshape(
            inp["fp8_pool"].shape[0], 2 * NKV, BLOCK, 128
        )
        try:
            _vendor_attention_args(
                kind, {**inp, "fp8_pool": blocked}, "fp8_pool", device
            )
            problems.append(f"{kind}: builder accepted the val7 blocked pool")
        except ValueError as exc:
            if "sparse_attn.py:24-27" not in str(exc):
                problems.append(
                    f"{kind}: blocked-pool refusal does not name the "
                    f"vendor layout: {exc}"
                )
        try:
            validate({**inp, "fp8_pool": blocked})
            problems.append(f"{kind}: validator accepted the val7 blocked pool")
        except ValueError as exc:
            if "VENDOR-layout" not in str(exc):
                problems.append(
                    f"{kind}: blocked-pool validator refusal unnamed: {exc}"
                )
        # The q4 container must never reach the original kernel: it is
        # tied to the 'pool' key, which only the overlay leg uses.
        try:
            _vendor_attention_args(
                kind, {**inp, "fp8_pool": inp["pool"]}, "fp8_pool", device
            )
            problems.append(f"{kind}: q4 container accepted under fp8_pool")
        except ValueError:
            pass

    audit("decode", hi.decode_inputs, hi.validate_decode_inputs)
    audit("prefill", hi.prefill_inputs, hi.validate_prefill_inputs)
    return {
        "passed": not problems,
        "problems": problems,
        "summary": (
            "the ONE vendor-invocation builder moves every argument, "
            "keeps the real impl's positional order, pins the fp8 unit "
            "scales, and refuses the val7 blocked pool"
            if not problems
            else "; ".join(problems)
        ),
    }


def _dry_decode_step_audit() -> dict:
    """Structure audit of the case-4 FULL decode step (no GPU on this
    VPS): the step inputs pass their anti-vacuous validator and every
    vacuity mutation fails closed; the one bundle mover puts EVERY step
    tensor on the simulated target with shapes/dtypes preserved; the
    output buffer and the production unit scales are allocated ON the
    target; and source pins prove the unit reaches the kernels ONLY
    through the overlay's public wrappers, restores the static buffers
    before every replay, compares every written buffer, and that the
    overlay's unit-scale gate is capture-safe (cached verdict, fail
    closed under capture)."""
    device = torch.device("meta")
    problems: list[str] = []
    inp = hi.decode_step_inputs()
    hi.validate_decode_step_inputs(inp)

    # (a) every mutation that would make the replay comparison vacuous
    #     (or replay an out-of-range index) fails closed, naming the
    #     problem.
    mutations = (
        ("new token on an unread page",
         lambda b: b.__setitem__("slot_mapping", torch.full_like(
             b["slot_mapping"], 8 * hi.BLOCK + 1))),
        ("top-k shifted off the new page",
         lambda b: b.__setitem__("topk", b["topk"] + 1)),
        ("positions beyond the rotary cache",
         lambda b: b.__setitem__(
             "positions", b["positions"] + hi.BLOCK)),
        ("index slot beyond the index cache",
         lambda b: b.__setitem__(
             "index_slot_mapping", b["index_slot_mapping"] + 8 * hi.BLOCK)),
        ("seq_lens shortened", lambda b: b.__setitem__(
            "seq_lens", b["seq_lens"] - hi.BLOCK)),
    )
    for label, mutate in mutations:
        bad = hi.decode_step_inputs()
        mutate(bad)
        try:
            hi.validate_decode_step_inputs(bad)
            problems.append(f"step validator accepted: {label}")
        except ValueError:
            pass

    # (b) mover coverage: nothing in the step's kernel-facing bundle may
    #     stay on the CPU (or silently change) when the bundle is moved.
    before = dict(_iter_tensors({k: inp[k] for k in _W1_STEP_KEYS}))
    moved = _kernel_bundle(inp, _W1_STEP_KEYS, device)
    after = dict(_iter_tensors(moved))
    if set(before) != set(after):
        problems.append("step: mover dropped/duplicated tensors")
    for path, b in before.items():
        a = after.get(path)
        if a is None:
            continue
        if a.device != device:
            problems.append(f"step/{path} stayed on {a.device}")
        if a.shape != b.shape or a.dtype != b.dtype:
            problems.append(f"step/{path} changed shape/dtype")

    # (c) the static bundle: the OUTPUT buffer is allocated on the
    #     target, the production unit scales are built ON the target
    #     (they are the tensors whose eager validation the capture must
    #     reuse), and the non-tensor sm_scale passes through unchanged.
    static = _decode_step_static_bundle(inp, device)
    if static["output"].device != device or (
        static["output"].shape != static["q_out"].shape
    ):
        problems.append("step: output buffer not allocated on the target")
    if static["unit_scale"].device != device:
        problems.append("step: unit scales not on the target device")
    if static["sm_scale"] != inp["sm_scale"]:
        problems.append("step: sm_scale passthrough broken")
    for path, t in _iter_tensors(static):
        if isinstance(t, torch.Tensor) and t.device != device:
            problems.append(f"step/{path} not on the target device")

    # (d) source pins: the launch goes through the overlay's PUBLIC
    #     wrappers only, and the unit restores the static buffers
    #     before every replay and compares every written buffer.
    harness_src = Path(__file__).read_text()

    def body(marker: str) -> str:
        start = harness_src.index(marker)
        return harness_src[start:harness_src.index("\ndef ", start + 1)]

    launch = body("def _launch_decode_step")
    if "_overlay_wrappers()" not in launch:
        problems.append("step launch does not use the overlay wrappers")
    if "q4k.q4_w1_insert" in launch or "q4k.q4_sparse_decode" in launch:
        problems.append("step launch bypasses the wrappers into the kernels")
    wrap = body("def _overlay_wrappers")
    for pin in ("_make_hook_wrapper", "__q4kv_original__"):
        if pin not in wrap:
            problems.append(f"wrapper construction lost the {pin} pin")
    unit4 = body("def unit_4")
    for pin in (
        "hi.validate_decode_step_inputs", "torch.cuda.graph",
        "set_state(pristine_qkv, pristine_pool)", "g.replay()",
        "_STEP_WRITE_KEYS", "mutation_live", "torch.roll",
    ):
        if pin not in unit4:
            problems.append(f"unit_4 lost the {pin!r} pin")

    # (e) source pins: the overlay's unit-scale gate is capture-safe
    #     (cached verdict bound to the exact tensor object, fail closed
    #     when first seen under capture) and both wrappers gate their
    #     scalar slots against tensor-coercion host syncs.
    src_root = Path(__file__).resolve().parents[1]
    overlay_src = (src_root / "src" / "vllm_exl3" / "minimax_q4kv.py").read_text()
    for pin in (
        "_stream_is_capturing", "weakref.ref(scale)",
        "first seen while a CUDA graph is",
        "cudagraph_utils.py:464-467",
    ):
        if pin not in overlay_src:
            problems.append(f"unit-scale gate lost the {pin!r} capture pin")
    w1_wrap = overlay_src[overlay_src.index("def _make_w1_wrapper"):]
    w1_wrap = w1_wrap[: w1_wrap.index("\ndef ")]
    import re as _re

    for pin in ("write_qkv", "eps", "skip_index_branch", "num_heads",
                "block_size"):
        if not _re.search(rf'_py_scalar\(\s*"{pin}"', w1_wrap):
            problems.append(f"W1 wrapper lost the {pin!r} scalar gate")
    dec_wrap = overlay_src[overlay_src.index("def _make_decode_wrapper"):]
    dec_wrap = dec_wrap[: dec_wrap.index("\ndef ")]
    for pin in ("sm_scale", "num_kv_heads", "decode_query_len"):
        if not _re.search(rf'_py_scalar\(\s*"{pin}"', dec_wrap):
            problems.append(f"decode wrapper lost the {pin!r} scalar gate")

    return {
        "passed": not problems,
        "problems": problems,
        "summary": (
            "the decode-step bundle moves whole, the output and unit "
            "scales land on the target, every vacuity mutation fails "
            "closed, and the wrappers' capture-safety pins are in place"
            if not problems
            else "; ".join(problems)
        ),
    }


def _dry_prefill_fast_audit() -> dict:
    """CPU audit of the fast prefill staging path (unit 6/7 structure).

    No triton, no GPU: builds the long-prefix fixture, checks the fast
    planner against the legacy planner at the timing lengths (the plan
    must be identical or the staged bytes differ), checks the fast
    dequant is bit-identical to the fp32 chain on real pool bytes, and
    pins the harness structure (both legs set the knob, both restore it,
    and unit 7's gates are the faster-than-legacy and 1.35x-of-no-staging
    pair on top of the hard staged-bytes gate). The knob defaults ON.
    """
    problems: list[str] = []
    # Production slice counts at the timing lengths (cap 192 pages from the
    # 48 MiB default, 128-token pages): 8K/16K fit one slice, 30K needs a
    # few. A fixture that explodes into ~191 slices (the old uniform-random
    # top-k) is not measuring the serve -- it staged 146x the prefix.
    for nb, want_slices in ((64, 1), (128, 1), (240, 3)):
        inp = hi.long_prefill_inputs(nb, chunk=CHUNK_TOKENS)
        cap = ref.staging_max_pages(NKV)
        legacy = ref.plan_prefill_slices(inp["pool"], inp["block_table"],
                                        inp["topk"], inp["cu_seqlens_q"],
                                        NKV, cap)
        fast = ref.plan_prefill_slices_fast(inp["pool"], inp["block_table"],
                                           inp["topk"], inp["cu_seqlens_q"],
                                           NKV, cap)
        if len(fast) != len(legacy):
            problems.append(f"{nb} pages: {len(fast)} != {len(legacy)} slices")
            continue
        if len(fast) != want_slices:
            problems.append(
                f"{nb} pages: {len(fast)} slices, production shape is "
                f"{want_slices} (fixture selection model drifted)"
            )
        staged = sum(s["n"] for s in fast)
        if not 1.0 <= staged / nb <= 3.0:
            problems.append(
                f"{nb} pages: staging amplification {staged / nb:.2f}x is "
                "not production-shaped (must stay a small multiple)"
            )
        for i, (a, b) in enumerate(zip(legacy, fast)):
            if a["q"] != b["q"] or not torch.equal(a["pages"], b["pages"]):
                problems.append(f"{nb} pages: plan differs at {a['q']}")
                break
            # every slice's staged bytes, over the exact staged region
            ba = torch.zeros((cap + 1, NKV, BLOCK, 2 * 128),
                             dtype=torch.bfloat16)
            bb = torch.zeros_like(ba)
            ref.fill_staging_buffer(ba, inp["pool"], a["pages"])
            ref.fill_staging_buffer_fast(bb, inp["pool"], b["pages"])
            if not torch.equal(ba[: b["n"] + 1], bb[: b["n"] + 1]):
                problems.append(f"{nb} pages: slice {i} staged bytes differ")
                break
        # the byte gate must not be vacuous: one flipped bf16 element in
        # one staged page has to be detected
        ba = torch.zeros((cap + 1, NKV, BLOCK, 2 * 128), dtype=torch.bfloat16)
        ref.fill_staging_buffer_fast(ba, inp["pool"], fast[0]["pages"])
        mutated = ba.clone()
        mutated[0, 0, 0, 0] = mutated[0, 0, 0, 0] + 1.0
        if torch.equal(ba[: fast[0]["n"] + 1], mutated[: fast[0]["n"] + 1]):
            problems.append(f"{nb} pages: the staged-byte gate is vacuous")
    # dequant bit-identity on the fixture's real packed bytes
    inp = hi.long_prefill_inputs(8, chunk=64)
    flat = inp["pool"].reshape(8, 2 * NKV, ref.SLOT_REGION)
    legacy_vals = ref.dequant_pool_pages(flat).to(torch.bfloat16)
    fast_vals = ref.dequant_pool_pages_bf16(flat)
    nan = torch.isnan(legacy_vals)
    if not torch.equal(nan, torch.isnan(fast_vals)) or not torch.equal(
        legacy_vals[~nan], fast_vals[~nan]
    ):
        problems.append("fast dequant is not bit-identical on fixture bytes")
    # an unvalidated fixture must fail closed
    bad = hi.long_prefill_inputs(8, chunk=64)
    bad["topk"] = bad["topk"].clone()
    bad["topk"][0, 0, 0] = 10_000
    try:
        hi.validate_long_prefill_inputs(bad)
    except ValueError:
        pass
    else:
        problems.append("out-of-range topk was not refused")
    src = Path(__file__).read_text()
    for pin in (
        'os.environ["VLLM_EXL3_Q4KV_PREFILL_FAST"] = "1"',
        'os.environ.pop("VLLM_EXL3_Q4KV_PREFILL_FAST", None)',
        '1.35 * top["layer_ms_no_staging"]',
        "fast_ms >= legacy_ms",
        "torch.equal(out_fast, out_legacy)",
        # unit 7 gates on STAGED BYTES, not on the attention output, and
        # carries a run-to-run determinism control for the vendor kernel
        "_staged_bytes_identical(inp, cap, \"cuda\")",
        "torch.equal(run_legacy(), run_legacy())",
        "vendor_output_deterministic",
        "vendor_outputs_identical",
    ):
        if pin not in src:
            problems.append(f"harness pin missing: {pin}")
    # the knob defaults ON, =0 is the explicit opt-out, and the wrapper
    # dispatches on it (fast pair by default, legacy pair under =0)
    import vllm_exl3.minimax_q4kv as q4ov

    saved = os.environ.pop("VLLM_EXL3_Q4KV_PREFILL_FAST", None)
    try:
        if q4ov._prefill_fast_enabled() is not True:
            problems.append("VLLM_EXL3_Q4KV_PREFILL_FAST is not default-on")
        os.environ["VLLM_EXL3_Q4KV_PREFILL_FAST"] = ""
        if q4ov._prefill_fast_enabled() is not True:
            problems.append("empty VLLM_EXL3_Q4KV_PREFILL_FAST is not on")
        os.environ["VLLM_EXL3_Q4KV_PREFILL_FAST"] = "0"
        if q4ov._prefill_fast_enabled() is not False:
            problems.append("VLLM_EXL3_Q4KV_PREFILL_FAST=0 is not the opt-out")
        os.environ["VLLM_EXL3_Q4KV_PREFILL_FAST"] = "maybe"
        try:
            q4ov._prefill_fast_enabled()
        except ValueError:
            pass
        else:
            problems.append("a garbage knob value did not fail closed")
    finally:
        if saved is not None:
            os.environ["VLLM_EXL3_Q4KV_PREFILL_FAST"] = saved
        else:
            os.environ.pop("VLLM_EXL3_Q4KV_PREFILL_FAST", None)
    for pin in (
        "plan = ref.plan_prefill_slices_fast",
        "fill = q4kernels.q4_stage_pages",
        "plan = ref.plan_prefill_slices",
        "fill = ref.fill_staging_buffer",
    ):
        if pin not in src:
            problems.append(f"wrapper dispatch pin missing: {pin}")
    return {
        "passed": not problems,
        "problems": problems,
        "summary": (
            "fast planner == legacy planner at 8K/16K/30K on a "
            "production-shaped fixture (1/1/3 slices, <=3x staged), every "
            "slice's staged bytes bit-identical over [:n+1], the byte gate "
            "is not vacuous, fast dequant bit-identical on real pool "
            "bytes, fixture fails closed, knob default-on with a =0 "
            "opt-out and a fail-closed garbage value, wrapper dispatch "
            "pinned, unit 6/7 gates pinned"
            if not problems
            else "; ".join(problems)
        ),
    }


def cpu_dry_run() -> int:
    units = [
        ("dry/0-w1-reference-runs", _dry_w1_reference),
        ("dry/1a-gate-numerics-selftest", _dry_gate_a),
        ("dry/1b-gate-bytes-selftest", _dry_gate_b),
        ("dry/1c-gate-index-codes-selftest", _dry_gate_c),
        ("dry/2-decode-reference-runs", _dry_decode),
        ("dry/3-pool-bytes", unit_3),
        ("dry/4-device-audit", _dry_device_audit),
        ("dry/5-prefill-staging", _dry_prefill),
        ("dry/6-vendor-invocation-structure", _dry_vendor_args_audit),
        ("dry/7-decode-step-structure", _dry_decode_step_audit),
        ("dry/8-prefill-fast-structure", _dry_prefill_fast_audit),
    ]
    return run_units(
        units,
        None,
        "cpu dry run: no GPU/triton required (reference paths and gate "
        "self-tests only)",
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _frontend_check_unit() -> dict:
    import q4kv_triton_frontend_check

    try:
        q4kv_triton_frontend_check.main()
    except SystemExit as exc:
        ok = exc.code in (0, None)
        return {
            "passed": ok,
            "summary": (
                "W1 + R2 frontend-compile for sm_121 in every variant"
                if ok
                else "frontend compile failure (see the errors above)"
            ),
        }
    return {"passed": True, "summary": "frontend check returned"}


def main() -> None:
    argv = list(sys.argv[1:])
    json_path: str | None = None
    dry = False
    it = iter(argv)
    for a in it:
        if a == "--cpu-dry-run":
            dry = True
        elif a == "--json":
            json_path = next(it, None)
            if json_path is None:
                sys.exit("--json requires a PATH argument")
        else:
            sys.exit(f"unknown argument {a} (only --json PATH and "
                     "--cpu-dry-run are supported)")
    if dry:
        sys.exit(cpu_dry_run())
    if not torch.cuda.is_available():
        sys.exit("GB10 only: CUDA is required (use --cpu-dry-run on the VPS)")
    cap = torch.cuda.get_device_capability()
    print(f"device: {torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}")
    units = [
        ("0/frontend-check", _frontend_check_unit),
        ("1a/w1-norm-rope-numerics", unit_1a),
        ("1b/w1-packing-bytes", unit_1b),
        ("1c/w1-index-bytes", unit_1c),
        ("1d/w1-rewrite-scope", unit_1d),
        ("2/r2-decode-accuracy", unit_2),
        ("3/pool-bytes", unit_3),
        ("4/cuda-graph-replay", unit_4),
        ("5/prefill-staging", unit_5),
        ("6/prefill-fast-staging", unit_6),
        ("7/prefill-timing", unit_7),
    ]
    sys.exit(
        run_units(
            units, json_path,
            f"GB10 q4 KV parity: {len(units)} units, all run to completion",
        )
    )


if __name__ == "__main__":
    main()
