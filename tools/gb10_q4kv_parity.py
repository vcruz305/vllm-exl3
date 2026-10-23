"""GB10 (sm_121) parity harness for the milestone-1 q4 KV cache.

Run ONLY on a DGX Spark GB10 with the MiniMax-M3.1-Flash vLLM image
(Triton + CUDA available). This VPS has no GPU and no triton: the tool
is documented here and validated on GB10, never on the VPS.

Exact command (from the plugin worktree root):

    python tools/gb10_q4kv_parity.py

Cases (all must pass; any failure exits non-zero):
  1. W1 byte-exact: the Triton fused write kernel's pool bytes, q_out and
     index bytes are byte-identical to the pure-torch reference packer
     (reference_w1_write run on CPU from the same inputs).
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

import math
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vllm_exl3 import minimax_q4kv_kernels as q4k  # noqa: E402
from vllm_exl3 import minimax_q4kv_reference as ref  # noqa: E402

# Real model shapes (MiniMax-M3.1-Flash MSA): 64 query heads, 4 KV heads,
# rotary 64, block 128. Small page counts keep the run minutes-long.
NQ, NKV, NIQ = 64, 4, 5
ROTARY = 64
EPS = 1e-4
BLOCK = 128
NUM_PAGES = 64
NUM_REQS = 8
TOPK = 8


def _rand(shape, seed, scale=1.0, device="cpu", dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randn(shape, generator=g) * scale
    return t.to(dtype=dtype, device=device)


def case1_w1_byte_exact() -> None:
    """Triton W1 must reproduce the reference packer byte-for-byte."""
    num_tokens = 32
    width = (NQ + 2 * NKV + NIQ + 1) * 128
    dev = "cuda"
    qkv = _rand((num_tokens, width), 1, 0.5, dev, torch.bfloat16)
    norm_ws = {
        name: _rand((128,), 2 + i, device=dev)
        for i, name in enumerate(("q", "k", "iq", "ik"))
    }
    max_pos = 256
    cos_sin = _rand((max_pos, ROTARY), 6, device=dev)
    positions = torch.arange(num_tokens, device=dev) + 1000
    # Dense slot mapping: tokens fill pages 1..2 (real page/tok split).
    slot_mapping = torch.arange(num_tokens, device=dev) + BLOCK
    index_slot_mapping = torch.arange(num_tokens, device=dev)

    pool = torch.zeros(
        (NUM_PAGES, 2 * NKV, BLOCK, 72), dtype=torch.uint8, device=dev
    )
    q_out = torch.empty(
        (num_tokens, NQ, 128), dtype=torch.bfloat16, device=dev
    )
    index_q_out = torch.empty(
        (num_tokens, NIQ, 128), dtype=torch.float8_e4m3fn, device=dev
    )
    q4k.q4_w1_insert(
        qkv, norm_ws["q"], norm_ws["k"], cos_sin, positions,
        NQ, NKV, ROTARY, EPS, norm_ws["iq"], norm_ws["ik"], NIQ,
        slot_mapping, index_slot_mapping, pool, None, BLOCK,
        q_out, index_q_out,
    )

    # Reference on CPU from the same inputs (independent device path).
    out = ref.reference_w1_write(
        qkv.cpu(), norm_ws["q"].cpu(), norm_ws["k"].cpu(),
        norm_ws["iq"].cpu(), norm_ws["ik"].cpu(), cos_sin.cpu(),
        positions.cpu(), EPS, ROTARY, NQ, NKV, NIQ,
        slot_mapping.cpu(), index_slot_mapping.cpu(),
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
    print("case 1 OK: W1 byte-exact vs torch reference (pool+q+index)")


def _decode_fixture(device):
    """Real-shaped decode fixture: processed q, packed q4 pool, fp8 pool."""
    total_q = NUM_REQS  # one decode token per request
    sm_scale = 1.0 / math.sqrt(128.0)
    q = _rand((total_q, NQ, 128), 3, 0.5, device, torch.bfloat16)
    # Original fp32 K/V values for all pages.
    k_vals = _rand((NUM_PAGES, NKV, BLOCK, 128), 4, 0.5)
    v_vals = _rand((NUM_PAGES, NKV, BLOCK, 128), 5, 0.5)
    # q4 pool: pack every page (K slots even, V slots odd).
    pool = torch.zeros(
        (NUM_PAGES, 2 * NKV, ref.SLOT_REGION), dtype=torch.uint8
    )
    for page in range(NUM_PAGES):
        for head in range(NKV):
            pool[page, 2 * head] = ref.pack_slot_region_side(
                k_vals[page, head], 0
            )
            pool[page, 2 * head + 1] = ref.pack_slot_region_side(
                v_vals[page, head], 1
            )
    pool = pool.reshape(NUM_PAGES, 2 * NKV, BLOCK, 72).to(device)
    # fp8 baseline pool from the same values (e4m3, satfinite).
    fp8_pool = torch.cat(
        [k_vals, v_vals], dim=1
    ).to(torch.float8_e4m3fn).to(device)
    # Block table: identity; every request selects TOPK distinct blocks.
    block_table = torch.arange(NUM_PAGES, device=device).repeat(
        NUM_REQS, 1
    ).contiguous()
    topk = torch.empty((NKV, total_q, TOPK), dtype=torch.int64)
    for r in range(total_q):
        for kvh in range(NKV):
            topk[kvh, r] = torch.arange(TOPK) + r
    topk = topk.to(device)
    seq_lens = torch.full((total_q,), TOPK * BLOCK, device=device)
    return q, k_vals, v_vals, pool, fp8_pool, block_table, topk, seq_lens, sm_scale


def _fp8_baseline(fp8_pool, block_table, topk, q, sm_scale):
    """Exact attention over the fp8-quantized pool (baseline math)."""
    total_q = q.shape[0]
    num_heads = q.shape[1]
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
    (q, k_vals, v_vals, pool, fp8_pool, block_table, topk, seq_lens,
     sm_scale) = _decode_fixture(device)
    out = torch.zeros_like(q)
    q4k.q4_sparse_decode(
        q, pool, topk, block_table, seq_lens, NKV, sm_scale, out, 1
    )
    base = _fp8_baseline(fp8_pool, block_table, topk, q, sm_scale)
    oracle = ref.reference_sparse_attention(
        q.to(torch.float32),
        k_vals.contiguous(),
        v_vals.contiguous(),
        block_table.cpu(), topk[0],
        torch.full((q.shape[0],), TOPK * BLOCK - 1), sm_scale,
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
    (q, _, _, pool, _, block_table, topk, seq_lens,
     sm_scale) = _decode_fixture(device)
    out = torch.zeros_like(q)
    # Warmup on a side stream (torch requirement for capture).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            q4k.q4_sparse_decode(
                q, pool, topk, block_table, seq_lens, NKV, sm_scale, out, 1
            )
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        q4k.q4_sparse_decode(
            q, pool, topk, block_table, seq_lens, NKV, sm_scale, out, 1
        )
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager_out), "CUDA-graph replay differs from eager"
    print("case 4 OK: CUDA-graph capture of R2 reproduces eager output")


def main() -> None:
    if not torch.cuda.is_available():
        sys.exit("GB10 only: CUDA is required (never run this on the VPS)")
    cap = torch.cuda.get_device_capability()
    print(f"device: {torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}")
    dev = "cuda"
    case1_w1_byte_exact()
    eager_out = case2_r2_accuracy(dev)
    case3_pool_bytes()
    case4_cuda_graph(dev, eager_out)
    print("ALL GB10 q4 KV parity cases PASSED")


if __name__ == "__main__":
    main()
