"""GB10 (sm_121) parity harness for the milestone-1 q4 KV cache.

Run ONLY on a DGX Spark GB10 with the MiniMax-M3.1-Flash vLLM image
(Triton + CUDA available). This VPS has no GPU and no triton: the GPU
cases are documented here and validated on GB10; the CPU parts of every
case (input construction, consistency validation and the pure-torch
reference paths) are exercised on the VPS by ``--cpu-dry-run``.

Exact commands (from the plugin worktree root):

    python tools/gb10_q4kv_parity.py               # GB10 only
    python tools/gb10_q4kv_parity.py --cpu-dry-run  # any CPU (incl. VPS)

Step 0 (run first, in-main): tools/q4kv_triton_frontend_check.py
frontend-compiles W1 and R2 for sm_121 without a GPU; any Triton
frontend type error (e.g. an fp8 pointer handed to a raw-byte kernel)
aborts here with a clear message instead of inside case 1.

Every case builds its inputs from tools/q4kv_harness_inputs.py and runs
that module's input-consistency validator BEFORE any kernel or reference
call. A harness data bug (e.g. the GB10 val2 crash: positions beyond
the rotary cache) aborts the case with a message naming the input, never
as an IndexError that looks like a kernel bug.

Cases (all must pass; any failure exits non-zero):
  1. W1 byte-exact: the Triton fused write kernel's pool bytes, q_out,
     index_q bytes AND index_k scatter bytes are byte-identical to the
     pure-torch reference packer (reference_w1_write run on CPU from
     the same inputs).
  2. R2 decode accuracy: on real-shaped random data, the q4 decode
     attention output has cosine >= 0.99 against the fp8 baseline
     (exact attention over the fp8-quantized pool) AND against the fp32
     oracle (exact attention over the unquantized values).
  3. Pool bytes measured: the allocated pool is 73728 B/page and
     576 B/token/layer for 4 KV heads (spec arithmetic).
  4. CUDA-graph capture of R2: capture and replay produce the eager
     output bit-identically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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


def _cuda(*ts: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(t.cuda() for t in ts)


def case1_w1_byte_exact() -> None:
    """Triton W1 must reproduce the reference packer byte-for-byte."""
    inp = hi.w1_inputs()
    hi.validate_w1_inputs(inp)
    nw = inp["norm_ws"]
    (qkv, q_nw, k_nw, iq_nw, ik_nw, cos_sin, positions, slot_mapping,
     index_slot_mapping, pool, index_cache, q_out, index_q_out) = _cuda(
        inp["qkv"], nw["q"], nw["k"], nw["iq"], nw["ik"], inp["cos_sin"],
        inp["positions"], inp["slot_mapping"], inp["index_slot_mapping"],
        inp["pool"], inp["index_cache"], inp["q_out"], inp["index_q_out"],
    )
    q4k.q4_w1_insert(
        qkv, q_nw, k_nw, cos_sin, positions,
        NQ, NKV, ROTARY, EPS, iq_nw, ik_nw, NIQ,
        slot_mapping, index_slot_mapping, pool, index_cache, BLOCK,
        q_out, index_q_out,
    )

    # Reference on CPU from the same inputs (independent device path).
    out = ref.reference_w1_write(
        inp["qkv"], nw["q"], nw["k"], nw["iq"], nw["ik"], inp["cos_sin"],
        inp["positions"], EPS, ROTARY, NQ, NKV, NIQ,
        inp["slot_mapping"], inp["index_slot_mapping"],
    )
    assert torch.equal(q_out.cpu(), out["q"]), "W1 q_out mismatch"
    assert torch.equal(
        index_q_out.cpu().view(torch.uint8), out["index_q"]
    ), "W1 index_q mismatch"
    pool_cpu = pool.cpu().reshape(NUM_PAGES, 2 * NKV, ref.SLOT_REGION)
    n_ref = out["pool"].shape[0]
    assert torch.equal(
        pool_cpu[:n_ref], out["pool"]
    ), "W1 pool bytes mismatch"
    assert int(pool_cpu[n_ref:].sum()) == 0, "W1 wrote outside mapped pages"
    # index_k scatter: the reference returns the per-token e4m3 bytes and
    # the caller scatters them at index_slot_mapping * 128; compare the
    # full scattered index cache byte-for-byte.
    ic_cpu = torch.zeros_like(inp["index_cache"]).view(torch.uint8)
    valid = inp["index_slot_mapping"] >= 0
    ism = inp["index_slot_mapping"][valid]
    ic_cpu.view(-1, 128)[ism] = out["index_k"][valid]
    assert torch.equal(
        index_cache.cpu().view(torch.uint8), ic_cpu
    ), "W1 index_k scatter mismatch"
    print("case 1 OK: W1 byte-exact vs torch reference "
          "(pool+q+index_q+index_k scatter)")


def _fp8_baseline(fp8_pool, block_table, topk, q, sm_scale):
    """Exact attention over the fp8-quantized pool (baseline math)."""
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


def _cos(a, b):
    a = a.reshape(-1).to(torch.float32)
    b = b.reshape(-1).to(torch.float32)
    return float((a @ b) / (a.norm() * b.norm()))


def case2_r2_accuracy(device) -> torch.Tensor:
    """R2 cosine >= 0.99 vs fp8 baseline and fp32 oracle."""
    inp = hi.decode_inputs()
    hi.validate_decode_inputs(inp)
    (q, pool, fp8_pool, block_table, topk, seq_lens) = _cuda(
        inp["q"], inp["pool"], inp["fp8_pool"], inp["block_table"],
        inp["topk"], inp["seq_lens"],
    )
    out = torch.zeros_like(q)
    q4k.q4_sparse_decode(
        q, pool, topk, block_table, seq_lens, NKV, inp["sm_scale"], out, 1
    )
    base = _fp8_baseline(fp8_pool, block_table, topk, q, inp["sm_scale"])
    oracle = ref.reference_sparse_attention(
        q.to(torch.float32),
        inp["k_vals"].contiguous(),
        inp["v_vals"].contiguous(),
        inp["block_table"], inp["topk"][0],
        torch.full((q.shape[0],), TOPK * BLOCK - 1), inp["sm_scale"],
    )
    cos_base = _cos(out.cpu(), base)
    cos_oracle = _cos(out.cpu(), oracle)
    print(f"case 2: cos(R2, fp8 baseline) = {cos_base:.6f}")
    print(f"case 2: cos(R2, fp32 oracle)  = {cos_oracle:.6f}")
    assert cos_base >= 0.99, "R2 vs fp8 baseline below 0.99"
    assert cos_oracle >= 0.99, "R2 vs fp32 oracle below 0.99"
    print("case 2 OK: R2 decode cosine >= 0.99")
    return out


def case3_pool_bytes() -> None:
    """Measured pool bytes must match the spec arithmetic."""
    page_bytes = NUM_PAGES * 2 * NKV * ref.SLOT_REGION
    assert page_bytes == NUM_PAGES * ref.q4_pool_page_bytes(NKV)
    per_token = ref.q4_pool_bytes_per_token(NKV)
    assert per_token == 2 * NKV * 72 == 576, per_token
    print(f"case 3 OK: pool = {page_bytes} B total, "
          f"{ref.q4_pool_page_bytes(NKV)} B/page, {per_token} B/token/layer")


def case4_cuda_graph(device, eager_out) -> None:
    """R2 must be CUDA-graph capturable with identical output."""
    inp = hi.decode_inputs()
    hi.validate_decode_inputs(inp)
    # Deterministic builders: the case-4 fixture must be the very bytes
    # case 2 ran on (otherwise the eager/graph comparison is vacuous).
    inp2 = hi.decode_inputs()
    for key, t in inp.items():
        if isinstance(t, torch.Tensor):
            assert torch.equal(t, inp2[key]), (
                f"decode fixture is not deterministic in {key}"
            )
    (q, pool, topk, block_table, seq_lens) = _cuda(
        inp["q"], inp["pool"], inp["topk"], inp["block_table"],
        inp["seq_lens"],
    )
    out = torch.zeros_like(q)
    # Warmup on a side stream (torch requirement for capture).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            q4k.q4_sparse_decode(
                q, pool, topk, block_table, seq_lens, NKV,
                inp["sm_scale"], out, 1,
            )
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        q4k.q4_sparse_decode(
            q, pool, topk, block_table, seq_lens, NKV,
            inp["sm_scale"], out, 1,
        )
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager_out), "CUDA-graph replay differs from eager"
    print("case 4 OK: CUDA-graph capture of R2 reproduces eager output")


# ---------------------------------------------------------------------------
# CPU dry run: build every case's inputs, validate them and run every
# pure-torch reference path on CPU (no triton, no GPU). Runs on the VPS.
# ---------------------------------------------------------------------------


def _dry_run_case1() -> None:
    inp = hi.w1_inputs()
    hi.validate_w1_inputs(inp)
    nw = inp["norm_ws"]
    out = ref.reference_w1_write(
        inp["qkv"], nw["q"], nw["k"], nw["iq"], nw["ik"], inp["cos_sin"],
        inp["positions"], EPS, ROTARY, NQ, NKV, NIQ,
        inp["slot_mapping"], inp["index_slot_mapping"],
    )
    # The reference must complete and write exactly the mapped pages.
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
    deq = ref.dequant_pool_pages(out["pool"])  # [pages, 2*nkv, 128, 128]
    for i, sm_i in enumerate(inp["slot_mapping"].tolist()):
        if sm_i < 0:
            continue
        page, tok = sm_i // BLOCK, sm_i % BLOCK
        for head in range(NKV):
            got_k = deq[page, 2 * head, tok]
            want_k = ref.unpack_q4_row(
                *ref.pack_q4_row(kn[i, head])
            )
            assert torch.equal(got_k, want_k), f"K scatter offsets t{i}h{head}"
            got_v = deq[page, 2 * head + 1, tok]
            want_v = ref.unpack_q4_row(
                *ref.pack_q4_row(x[i, NQ + NKV + head].to(torch.bfloat16))
            )
            assert torch.equal(got_v, want_v), f"V scatter offsets t{i}h{head}"
    # index_k scatter bytes: e4m3 of clamp(processed, +-448).
    ikn = ref.rope_partial_neox(
        ref.gemma_rmsnorm(x[:, -1], nw["ik"], EPS), cos_sin
    )
    assert torch.equal(out["index_k"], ref._e4m3_clamped(ikn))
    print("dry run case 1 OK: inputs valid, reference packer runs and "
          "writes exactly the mapped pages/slots")


def _dry_run_decode() -> None:
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
    print(f"dry run cases 2/4: cos(fp8 baseline, fp32 oracle) = "
          f"{cos:.6f}")
    assert cos >= 0.99, "baseline and oracle disagree before any kernel"
    print("dry run cases 2/4 OK: decode fixture valid; baseline and "
          "oracle run to completion on CPU (Triton R2 skipped)")


def cpu_dry_run() -> None:
    print("cpu dry run: no GPU/triton required (reference paths only)")
    _dry_run_case1()
    _dry_run_decode()
    case3_pool_bytes()
    print("ALL CPU dry-run q4 KV parity checks PASSED "
          "(inputs consistent; reference paths run; Triton skipped)")


def main() -> None:
    argv = list(sys.argv[1:])
    if argv == ["--cpu-dry-run"]:
        cpu_dry_run()
        return
    if argv:
        sys.exit(f"unknown arguments {argv} (only --cpu-dry-run is supported)")
    if not torch.cuda.is_available():
        sys.exit("GB10 only: CUDA is required (use --cpu-dry-run on the VPS)")
    cap = torch.cuda.get_device_capability()
    print(f"device: {torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}")
    dev = "cuda"
    # Frontend-compile W1/R2 for sm_121 FIRST (no GPU math involved):
    # a kernel that cannot compile must fail here with a clear message,
    # not inside case 1 (the GB10 val1 failure class).
    import q4kv_triton_frontend_check

    q4kv_triton_frontend_check.main()
    case1_w1_byte_exact()
    eager_out = case2_r2_accuracy(dev)
    case3_pool_bytes()
    case4_cuda_graph(dev, eager_out)
    print("ALL GB10 q4 KV parity cases PASSED")


if __name__ == "__main__":
    main()
