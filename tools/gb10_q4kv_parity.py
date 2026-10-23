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
  2.  R2 decode accuracy: on real-shaped random data, the q4 decode
      attention output has cosine >= 0.99 against the fp8 baseline AND
      against the fp32 oracle, with no NaN/Inf.
  3.  Pool bytes: the allocated pool is 73728 B/page and 576
      B/token/layer for 4 KV heads (spec arithmetic).
  4.  CUDA-graph capture of R2 under the static-buffer rules: every
      input is a preallocated CUDA tensor that is copied into before
      replay; the replay must reproduce the eager output bit-identically
      AND must read the CURRENT bytes of the static buffers (verified
      by copying new pool bytes in between replays).

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
import sys
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
# bundle, where the pure-torch oracles consume it.
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
    values = fp8_pool.view(NUM_PAGES, 2 * NKV, BLOCK, 128).to(torch.float32)
    # The fp8 pool is blocked [K | V], NOT interleaved like the q4 slots.
    k_pages = values[:, :NKV].contiguous()
    v_pages = values[:, NKV:].contiguous()
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


def unit_2() -> dict:
    """R2 decode cosine >= 0.99 vs fp8 baseline and fp32 oracle."""
    inp = _decode_fixture()
    gpu = _r2_static_bundle(inp, "cuda")
    out = _run_r2_bundle(gpu)
    base = _fp8_baseline(
        inp["fp8_pool"], inp["block_table"], inp["topk"], inp["q"],
        inp["sm_scale"],
    )
    oracle = ref.reference_sparse_attention(
        inp["q"].to(torch.float32),
        inp["k_vals"].contiguous(),
        inp["v_vals"].contiguous(),
        inp["block_table"], inp["topk"][0],
        torch.full((inp["q"].shape[0],), TOPK * BLOCK - 1),
        inp["sm_scale"],
    )
    cos_base = _cos(out, base)
    cos_oracle = _cos(out, oracle)
    finite = bool(torch.isfinite(out).all())
    passed = finite and cos_base >= 0.99 and cos_oracle >= 0.99
    return {
        "passed": passed,
        "cos_vs_fp8_baseline": cos_base,
        "cos_vs_fp32_oracle": cos_oracle,
        "output_finite": finite,
        "summary": (
            f"cos(R2, fp8 baseline)={cos_base:.6f} "
            f"cos(R2, fp32 oracle)={cos_oracle:.6f} finite={finite}"
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
    """R2 must be CUDA-graph capturable, and the replay must read the
    CURRENT bytes of the static input buffers.

    Static-buffer rules: every input is a preallocated CUDA tensor that
    is copied into before replay. The check is two-stage: (a) replay on
    the captured fixture bytes must reproduce the eager output
    bit-identically, and (b) after NEW pool bytes are copied into the
    static buffer, the replay must reproduce the eager run on THOSE
    bytes — a stale replay of the captured bytes fails (b) because the
    outputs differ.
    """
    inp = _decode_fixture()
    gpu = _r2_static_bundle(inp, "cuda")
    eager_fixture = _run_r2_bundle(gpu)
    # Warmup on a side stream (torch requirement for capture).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _launch_r2(gpu)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _launch_r2(gpu)
    # Stage (a): replay on the captured bytes == eager on those bytes.
    gpu["output"].zero_()
    g.replay()
    torch.cuda.synchronize()
    replay_fixture = gpu["output"].cpu()
    stage_a = bool(torch.equal(replay_fixture, eager_fixture))
    # Stage (b): copy NEW bytes into the static pool buffer, then replay
    # must match the eager run on the NEW bytes (whole pages rotated, so
    # every K/V value moves to a different block).
    gpu["pool"].copy_(torch.roll(gpu["pool"], shifts=3, dims=0))
    eager_mutated = _run_r2_bundle(gpu)
    gpu["output"].zero_()
    g.replay()
    torch.cuda.synchronize()
    replay_mutated = gpu["output"].cpu()
    stage_b = bool(torch.equal(replay_mutated, eager_mutated))
    mutation_live = not bool(torch.equal(eager_mutated, eager_fixture))
    passed = stage_a and stage_b and mutation_live
    if passed:
        summary = (
            "replay reproduces the eager output bit-identically and reads "
            "the current static-buffer bytes"
        )
    elif not stage_a:
        summary = "replay on the captured bytes differs from eager"
    elif not mutation_live:
        summary = "static-buffer mutation vacuous (outputs unchanged)"
    else:
        summary = "replay did not read the mutated static-buffer bytes"
    return {"passed": passed, "summary": summary}


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


def cpu_dry_run() -> int:
    units = [
        ("dry/0-w1-reference-runs", _dry_w1_reference),
        ("dry/1a-gate-numerics-selftest", _dry_gate_a),
        ("dry/1b-gate-bytes-selftest", _dry_gate_b),
        ("dry/1c-gate-index-codes-selftest", _dry_gate_c),
        ("dry/2-decode-reference-runs", _dry_decode),
        ("dry/3-pool-bytes", unit_3),
        ("dry/4-device-audit", _dry_device_audit),
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
    ]
    sys.exit(
        run_units(
            units, json_path,
            f"GB10 q4 KV parity: {len(units)} units, all run to completion",
        )
    )


if __name__ == "__main__":
    main()
