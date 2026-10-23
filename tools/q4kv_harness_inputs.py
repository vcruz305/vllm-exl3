"""Shared input builders and consistency validators for the q4 KV harnesses.

Both the GB10 parity harness (tools/gb10_q4kv_parity.py, GPU) and its
--cpu-dry-run mode build their inputs HERE, and every case validates its
inputs with the validators in this module BEFORE any kernel runs. The
GB10 val2 crash (an IndexError deep inside the reference because the
harness generated positions >= 1000 with a 256-row cos_sin cache) must
never again look like a kernel bug: a data inconsistency now aborts the
case with a message that says exactly which input is out of range.

This module is pure torch (no triton, no vllm) and runs on the CPU-only
red venv. The rotary cache is the real one, not random noise: it is built
with the vLLM RotaryEmbedding formula (NeoX halved
``[cos(pos * inv_freq) | sin(pos * inv_freq)]`` rows) using the
MiniMax-M3 partial-RoPE configuration (rope_theta 5e6, rotary_dim 64 =
partial_rotary_factor 0.5 of head_dim 128, as in the model config the
overlay serves), sized to cover max(positions) + 1 rows.
"""

from __future__ import annotations

import math

import torch

from vllm_exl3 import minimax_q4kv_reference as ref

# Real model shapes (MiniMax-M3.1-Flash MSA): 64 query heads, 4 KV heads,
# 5 indexer heads, rotary 64, block 128. Small page counts keep runs short.
NQ, NKV, NIQ = 64, 4, 5
ROTARY = 64
EPS = 1e-4
BLOCK = 128
NUM_PAGES = 64
NUM_REQS = 8
TOPK = 8
# MiniMax-M3 partial RoPE: rope_theta 5e6 and rotary_dim 64 of head_dim 128
# (partial_rotary_factor 0.5), matching the model configuration the q4
# overlay serves; the cache layout is vLLM's NeoX halved form.
ROPE_THETA = 5_000_000.0


def _rand(shape, seed, scale=1.0, dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(shape, generator=g) * scale).to(dtype)


def _integral(t: torch.Tensor) -> bool:
    return t.dtype in (torch.int32, torch.int64, torch.int16, torch.int8)


def build_cos_sin_cache(
    max_pos: int,
    rotary_dim: int = ROTARY,
    theta: float = ROPE_THETA,
) -> torch.Tensor:
    """The real rotary cache, as vLLM's RotaryEmbedding builds it.

    Row ``p`` is the NeoX halved form ``[cos(p * inv_freq) | sin(p *
    inv_freq)]`` with ``inv_freq_j = theta ** (-2j / rotary_dim)``; the
    W1 kernel and the reference both index it as
    ``cache[position, :rotary_dim]``.
    """
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim
        )
    )
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()


# ---------------------------------------------------------------------------
# Case 1 (W1 byte-exact): input builder + validator
# ---------------------------------------------------------------------------


def w1_inputs() -> dict:
    """Every case-1 input, built on CPU (the harness moves them to CUDA).

    The slot mappings deliberately cross a page boundary (slots
    112..143 span pool pages 0 and 1, and index-cache pages 0 and 1) so
    the page-split arithmetic is exercised on both sides of the boundary.
    """
    num_tokens = 32
    width = (NQ + 2 * NKV + NIQ + 1) * 128
    qkv = _rand((num_tokens, width), 1, 0.5, torch.bfloat16)
    norm_ws = {
        name: _rand((128,), 2 + i)
        for i, name in enumerate(("q", "k", "iq", "ik"))
    }
    positions = torch.arange(num_tokens, dtype=torch.int64) + 1000
    cos_sin = build_cos_sin_cache(int(positions.max()) + 1)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64) + (
        BLOCK - 16
    )  # 112..143 -> pool pages 0 and 1
    index_slot_mapping = slot_mapping.clone()  # -> index pages 0 and 1
    pool = torch.zeros(
        (NUM_PAGES, 2 * NKV, BLOCK, 72), dtype=torch.uint8
    )
    q_out = torch.empty(
        (num_tokens, NQ, 128), dtype=torch.bfloat16
    )
    index_q_out = torch.empty(
        (num_tokens, NIQ, 128), dtype=torch.float8_e4m3fn
    )
    # Real fp8 e4m3 index cache: the W1 kernel scatters index_k bytes into
    # it at index_slot_mapping * 128. Passing None here (the val2 harness
    # bug) makes the wrapper hand the POOL as the placeholder, so the
    # index_k scatter would corrupt pool bytes and break byte-exactness.
    index_cache = torch.zeros(
        (2, BLOCK, 128), dtype=torch.float8_e4m3fn
    )
    return dict(
        qkv=qkv, norm_ws=norm_ws, positions=positions, cos_sin=cos_sin,
        slot_mapping=slot_mapping, index_slot_mapping=index_slot_mapping,
        pool=pool, q_out=q_out, index_q_out=index_q_out,
        index_cache=index_cache, eps=EPS, rotary_dim=ROTARY,
    )


def validate_w1_inputs(inp: dict) -> None:
    """Fail closed on any shape/range inconsistency in the case-1 inputs.

    Raises ValueError with one message per problem, before any kernel or
    reference runs, so a harness data bug is never misread as a kernel
    numerics bug.
    """
    errors: list[str] = []
    n = inp["qkv"].shape[0]
    width = (NQ + 2 * NKV + NIQ + 1) * 128
    if inp["qkv"].shape != (n, width) or inp["qkv"].dtype != torch.bfloat16:
        errors.append(
            f"qkv must be [{n}, {width}] bfloat16, got "
            f"{tuple(inp['qkv'].shape)} {inp['qkv'].dtype}"
        )
    if n <= 0:
        errors.append("qkv must carry at least one token")
    for name in ("q", "k", "iq", "ik"):
        w = inp["norm_ws"][name]
        if w.shape != (128,) or not w.is_floating_point():
            errors.append(
                f"{name}_norm_weight must be [128] float, got "
                f"{tuple(w.shape)} {w.dtype}"
            )
    pos = inp["positions"]
    if pos.shape != (n,) or not _integral(pos):
        errors.append(
            f"positions must be [{n}] integral, got "
            f"{tuple(pos.shape)} {pos.dtype}"
        )
    else:
        if int(pos.min()) < 0:
            errors.append(f"positions must be >= 0, got {int(pos.min())}")
        # The GB10 val2 crash class: positions beyond the rotary cache.
        rows = inp["cos_sin"].shape[0]
        if int(pos.max()) >= rows:
            errors.append(
                f"positions up to {int(pos.max())} exceed the cos_sin cache "
                f"({rows} rows): build_cos_sin_cache must be sized to "
                f"max(positions) + 1"
            )
    cs = inp["cos_sin"]
    if cs.dim() != 2 or cs.shape[-1] != ROTARY:
        errors.append(
            f"cos_sin cache must be [rows, {ROTARY}], got "
            f"{tuple(cs.shape)}"
        )
    sm = inp["slot_mapping"]
    if sm.shape != (n,) or not _integral(sm):
        errors.append(
            f"slot_mapping must be [{n}] integral, got "
            f"{tuple(sm.shape)} {sm.dtype}"
        )
    else:
        mx = int(sm.max())
        if mx >= NUM_PAGES * BLOCK or int(sm.min()) < -1:
            errors.append(
                f"slot_mapping in [-1, {NUM_PAGES * BLOCK}) required, got "
                f"[{int(sm.min())}, {mx}] against {NUM_PAGES} pool pages"
            )
    ism = inp["index_slot_mapping"]
    if not isinstance(inp["index_cache"], torch.Tensor):
        errors.append(
            "index_cache must be a real fp8 e4m3 buffer (None makes the "
            "wrapper hand the pool as the scatter placeholder and corrupts "
            f"pool bytes), got {inp['index_cache']!r}"
        )
        ic_pages = 0
    else:
        if inp["index_cache"].dtype != torch.float8_e4m3fn:
            errors.append(
                "index_cache must be a real fp8 e4m3 buffer (None makes "
                "the wrapper hand the pool as the scatter placeholder and "
                f"corrupts pool bytes), got {inp['index_cache'].dtype}"
            )
        ic_pages = inp["index_cache"].shape[0]
    if ism.shape != (n,) or not _integral(ism):
        errors.append(
            f"index_slot_mapping must be [{n}] integral, got "
            f"{tuple(ism.shape)} {ism.dtype}"
        )
    elif ic_pages:
        mx = int(ism.max())
        if mx >= ic_pages * BLOCK or int(ism.min()) < -1:
            errors.append(
                f"index_slot_mapping in [-1, {ic_pages * BLOCK}) required, "
                f"got [{int(ism.min())}, {mx}] against {ic_pages} index "
                f"cache pages"
            )
    pool = inp["pool"]
    if (
        pool.shape != (NUM_PAGES, 2 * NKV, BLOCK, 72)
        or pool.dtype != torch.uint8
    ):
        errors.append(
            f"pool must be [{NUM_PAGES}, {2 * NKV}, {BLOCK}, 72] uint8, "
            f"got {tuple(pool.shape)} {pool.dtype}"
        )
    elif (
        pool.stride(2) != 72 or pool.stride(1) != 9216 or pool.stride(3) != 1
    ):
        errors.append("pool slot regions must be contiguous")
    q_out = inp["q_out"]
    if q_out.shape != (n, NQ, 128) or q_out.dtype != torch.bfloat16:
        errors.append(
            f"q_out must be [{n}, {NQ}, 128] bfloat16, got "
            f"{tuple(q_out.shape)} {q_out.dtype}"
        )
    iq_out = inp["index_q_out"]
    if iq_out.shape != (n, NIQ, 128) or iq_out.dtype != torch.float8_e4m3fn:
        errors.append(
            f"index_q_out must be [{n}, {NIQ}, 128] float8_e4m3fn, got "
            f"{tuple(iq_out.shape)} {iq_out.dtype}"
        )
    if errors:
        raise ValueError(
            "q4 KV harness case-1 input data inconsistent (fix the harness "
            "data, not the kernel): " + "; ".join(errors)
        )


# ---------------------------------------------------------------------------
# Cases 2 and 4 (R2 decode): input builder + validator
# ---------------------------------------------------------------------------


def decode_inputs() -> dict:
    """Every case-2/case-4 input, built on CPU (the harness moves them)."""
    sm_scale = 1.0 / math.sqrt(128.0)
    q = _rand((NUM_REQS, NQ, 128), 3, 0.5, torch.bfloat16)
    k_vals = _rand((NUM_PAGES, NKV, BLOCK, 128), 4, 0.5)
    v_vals = _rand((NUM_PAGES, NKV, BLOCK, 128), 5, 0.5)
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
    pool = pool.reshape(NUM_PAGES, 2 * NKV, BLOCK, 72)
    # The fp8/bf16 pools are in the VENDOR layout the ORIGINAL kernels
    # read and the W1 op writes: [pages, num_kv_heads, 128, 2*head_dim]
    # with [K | V] in every token row (common/ops/sparse_attn.py:24-27
    # and the V load at :206-215; csrc
    # fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu:890; the
    # allocator's per-token row is (head_size + head_size_v) wide,
    # v1/kv_cache_interface.py:557). The GB10 val7 harness handed the
    # original kernel a K/V-BLOCKED pool ([pages, 2*num_kv_heads, 128,
    # 128]) instead and the fp8 leg came back uncorrelated (cos 0.0022):
    # the kernel's V pointer landed one token row over, inside the K
    # data. bf16_pool is the same K/V with NO quantization -- the ground
    # truth the vendor kernel itself computes.
    fp8_pool = torch.cat([k_vals, v_vals], dim=-1).to(torch.float8_e4m3fn)
    bf16_pool = torch.cat([k_vals, v_vals], dim=-1).to(torch.bfloat16)
    block_table = (
        torch.arange(NUM_PAGES, dtype=torch.int64)
        .repeat(NUM_REQS, 1)
        .contiguous()
    )
    topk = torch.empty((NKV, NUM_REQS, TOPK), dtype=torch.int64)
    for r in range(NUM_REQS):
        for kvh in range(NKV):
            topk[kvh, r] = torch.arange(TOPK) + r
    seq_lens = torch.full((NUM_REQS,), TOPK * BLOCK, dtype=torch.int64)
    return dict(
        q=q, k_vals=k_vals, v_vals=v_vals, pool=pool, fp8_pool=fp8_pool,
        bf16_pool=bf16_pool, block_table=block_table, topk=topk,
        seq_lens=seq_lens, sm_scale=sm_scale,
    )


def _validate_vendor_kv_pools(inp: dict, pages: int, errors: list) -> None:
    """Fail closed unless the fp8/bf16 pools are the vendor-layout
    encodings of the fixture's own k_vals/v_vals.

    The ORIGINAL kernels read -- and the W1 op writes -- the pool as
    [pages, num_kv_heads, 128, 2*head_dim] with [K | V] in every token
    row (common/ops/sparse_attn.py:24-27 and the V load at :206-215;
    csrc fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu:890; the
    allocator's per-token row is (head_size + head_size_v) wide,
    v1/kv_cache_interface.py:557). The GB10 val7 harness built a
    K/V-BLOCKED pool instead ([pages, 2*num_kv_heads, 128, 128]) and the
    fp8 leg came back uncorrelated (cos 0.0022): the kernel's V pointer
    landed one token row over, inside the K data. Any other geometry --
    including that one -- must fail closed here.
    """
    vendor_shape = (pages, NKV, BLOCK, 256)
    fp8_pool = inp.get("fp8_pool")
    if (
        not isinstance(fp8_pool, torch.Tensor)
        or fp8_pool.shape != vendor_shape
        or fp8_pool.dtype != torch.float8_e4m3fn
    ):
        errors.append(
            "fp8 baseline pool must be the VENDOR-layout e4m3 encoding of "
            f"the same K/V: [{pages}, {NKV}, {BLOCK}, 256] with [K | V] in "
            f"every token row, got {tuple(fp8_pool.shape) if isinstance(fp8_pool, torch.Tensor) else fp8_pool} "
            f"{fp8_pool.dtype if isinstance(fp8_pool, torch.Tensor) else type(fp8_pool).__name__}"
        )
    elif not (
        torch.equal(
            fp8_pool.view(torch.uint8)[..., :128],
            inp["k_vals"].to(torch.float8_e4m3fn).view(torch.uint8),
        )
        and torch.equal(
            fp8_pool.view(torch.uint8)[..., 128:],
            inp["v_vals"].to(torch.float8_e4m3fn).view(torch.uint8),
        )
    ):
        errors.append("fp8 pool halves must be the e4m3 encodings of k_vals/v_vals")
    bf16_pool = inp.get("bf16_pool")
    if (
        not isinstance(bf16_pool, torch.Tensor)
        or bf16_pool.shape != vendor_shape
        or bf16_pool.dtype != torch.bfloat16
    ):
        errors.append(
            "bf16 ground-truth pool must be the same K/V unquantized in "
            f"the vendor layout [{pages}, {NKV}, {BLOCK}, 256] bfloat16"
        )
    elif not (
        torch.equal(bf16_pool[..., :128], inp["k_vals"].to(torch.bfloat16))
        and torch.equal(bf16_pool[..., 128:], inp["v_vals"].to(torch.bfloat16))
    ):
        errors.append(
            "bf16 pool halves must be k_vals/v_vals unquantized"
        )


def validate_decode_inputs(inp: dict) -> None:
    """Fail closed on any shape/range inconsistency in the R2 fixture."""
    errors: list[str] = []
    q = inp["q"]
    if q.shape != (NUM_REQS, NQ, 128) or q.dtype != torch.bfloat16:
        errors.append(
            f"q must be [{NUM_REQS}, {NQ}, 128] bfloat16, got "
            f"{tuple(q.shape)} {q.dtype}"
        )
    pool = inp["pool"]
    if (
        pool.shape != (NUM_PAGES, 2 * NKV, BLOCK, 72)
        or pool.dtype != torch.uint8
    ):
        errors.append(
            f"pool must be [{NUM_PAGES}, {2 * NKV}, {BLOCK}, 72] uint8, got "
            f"{tuple(pool.shape)} {pool.dtype}"
        )
    elif (
        pool.stride(2) != 72 or pool.stride(1) != 9216 or pool.stride(3) != 1
    ):
        errors.append("pool slot regions must be contiguous")
    _validate_vendor_kv_pools(inp, NUM_PAGES, errors)
    bt = inp["block_table"]
    max_blocks = bt.shape[1] if bt.dim() == 2 else -1
    if bt.dim() != 2 or bt.shape[0] != NUM_REQS or not _integral(bt):
        errors.append(
            f"block_table must be [{NUM_REQS}, n_blocks] integral, got "
            f"{tuple(bt.shape)} {bt.dtype}"
        )
    else:
        lo, hi = int(bt.min()), int(bt.max())
        if lo < -1 or hi >= NUM_PAGES:
            errors.append(
                f"block_table entries must be physical pool pages in "
                f"[-1, {NUM_PAGES}), got [{lo}, {hi}]"
            )
    topk = inp["topk"]
    if topk.shape != (NKV, NUM_REQS, TOPK) or not _integral(topk):
        errors.append(
            f"topk must be [{NKV}, {NUM_REQS}, {TOPK}] integral, got "
            f"{tuple(topk.shape)} {topk.dtype}"
        )
    elif max_blocks > 0:
        lo, hi = int(topk.min()), int(topk.max())
        if lo < -1 or hi >= max_blocks:
            errors.append(
                f"topk entries are logical block-table indices and must be "
                f"in [-1, {max_blocks}), got [{lo}, {hi}]"
            )
    seq = inp["seq_lens"]
    if seq.shape != (NUM_REQS,) or not _integral(seq):
        errors.append(
            f"seq_lens must be [{NUM_REQS}] integral, got "
            f"{tuple(seq.shape)} {seq.dtype}"
        )
    else:
        if int(seq.max()) > max_blocks * BLOCK:
            errors.append(
                f"seq_lens up to {int(seq.max())} exceed the block table's "
                f"{max_blocks} x {BLOCK} accessible tokens"
            )
        # Fixture invariant: seq_lens == TOPK * BLOCK pins the kernel's
        # n_blocks mask, the fp8 baseline's causal mask and the fp32
        # oracle's query_pos to the same token set.
        if int(seq.min()) != TOPK * BLOCK or int(seq.max()) != TOPK * BLOCK:
            errors.append(
                f"fixture seq_lens must be exactly TOPK * BLOCK = "
                f"{TOPK * BLOCK}, got [{int(seq.min())}, {int(seq.max())}]"
            )
    sm_scale = inp["sm_scale"]
    if not (sm_scale > 0 and math.isfinite(sm_scale)):
        errors.append(f"sm_scale must be positive and finite, got {sm_scale}")
    if errors:
        raise ValueError(
            "q4 KV harness decode input data inconsistent (fix the "
            "harness data, not the kernel): " + "; ".join(errors)
        )


# ---------------------------------------------------------------------------
# Case 5 (prefill staging): input builder + validator
# ---------------------------------------------------------------------------

# Small enough for the CPU fp32 oracle, large enough to force page-capped
# slicing under the harness cap override (2 MiB = 8 pages): sequence 0's
# cached prefix makes its causal valid-block count reach deep into its
# own page span, so its per-query unions exceed the cap and it is
# query-split; sequences 1 and 2 group under the cap.
PREFILL_NUM_PAGES = 32
PREFILL_MAX_BLOCKS = 12  # logical blocks per block-table row
PREFILL_TOPK = 8
PREFILL_QLENS = (8, 6, 4)  # three sequences of different lengths
PREFILL_PREFIX = (1024, 256, 0)  # cached context per sequence
# The R2 gate convention: the pinned 0.99 cosine floor is calibrated on
# unit-ish K/V magnitudes (RMS ~0.5). The W1-realistic structure (Gemma
# norm + partial RoPE) is kept, the post-W1 rows are scaled to the R2
# magnitude so the prefill gate measures the same quantization-noise
# regime as the shipped R2 gate (at raw W1 magnitudes, RMS ~2, the
# softmax is much sharper and the q4-vs-oracle cosine floor is ~0.98 for
# quantization noise alone -- a property of 4-bit KV, not of staging).
_W1P_VALUE_SCALE = 0.25  # post-norm RMS ~2 -> ~0.5

_W1P_NORM_W_SEED = 21


def prefill_inputs() -> dict:
    """Every case-5 input, built on CPU (the harness moves them to CUDA).

    Realism per the GB10 boot18 prefill crash: the K/V pool bytes are the
    W1 reference output over a REAL partial-RoPE table (theta 5e6,
    rotary_dim 64 of head_dim 128, exactly as build_cos_sin_cache builds
    for W1), the index cache is the real fp8 e4m3 encoding of the W1
    index branch, the fp8 baseline pool is the real e4m3 encoding of the
    SAME post-W1 K/V (what the vendor fp8 KV path would store), and the
    top-k is per (kv head, query): every entry below the causal
    valid-block count is a distinct valid logical block, everything above
    is the -1 sentinel, exactly as the indexer emits. The block-table
    rows share a physical prefix (pages 0..3) and own distinct pages
    afterwards, so slices overlap and re-select pages across boundaries.
    """
    nkv = NKV
    pages = PREFILL_NUM_PAGES
    # --- real partial-RoPE table over every pool-token position ---
    positions = torch.arange(pages * BLOCK, dtype=torch.int64)
    cos_sin = build_cos_sin_cache(int(positions.max()) + 1)
    cs_tok = cos_sin[positions]  # [pages*BLOCK, ROTARY]
    g = torch.Generator().manual_seed(_W1P_NORM_W_SEED)
    norm_ws = {
        name: 1.0 + 0.05 * torch.randn((128,), generator=g)
        for _, name in enumerate(("q", "k", "iq", "ik"))
    }
    x = torch.randn((pages * BLOCK, 2 * nkv, 128), generator=g) * 0.5
    # K side: the real W1 processing (Gemma norm + partial NeoX RoPE),
    # scaled to the R2 gate magnitude convention (see _W1P_VALUE_SCALE).
    k_tok = _W1P_VALUE_SCALE * ref.rope_partial_neox(
        ref.gemma_rmsnorm(x[:, :nkv], norm_ws["k"], EPS), cs_tok
    )
    # V side: V is not normed (W1 writes V through unchanged).
    v_tok = x[:, nkv:]
    # Index branch: the real fp8 e4m3 index-K bytes of the W1 output.
    ik_tok = ref._e4m3_clamped(
        ref.rope_partial_neox(
            ref.gemma_rmsnorm(
                torch.randn((pages * BLOCK, 128), generator=g) * 0.5,
                norm_ws["ik"],
                EPS,
            ),
            cs_tok,
        )
    )
    index_cache = ik_tok.view(pages * BLOCK, 128).to(torch.float8_e4m3fn)
    # --- pack the pool (the q4 bytes of the same K/V) ---
    k_vals = k_tok.view(pages, BLOCK, nkv, 128).permute(0, 2, 1, 3).contiguous()
    v_vals = v_tok.view(pages, BLOCK, nkv, 128).permute(0, 2, 1, 3).contiguous()
    flat = torch.zeros((pages, 2 * nkv, ref.SLOT_REGION), dtype=torch.uint8)
    for page in range(pages):
        for head in range(nkv):
            flat[page, 2 * head] = ref.pack_slot_region_side(
                k_vals[page, head], 0
            )
            flat[page, 2 * head + 1] = ref.pack_slot_region_side(
                v_vals[page, head], 1
            )
    pool = flat.reshape(pages, 2 * nkv, BLOCK, 72)
    # fp8_pool is the real e4m3 encoding of the SAME post-W1 K/V in the
    # VENDOR layout (see decode_inputs); bf16_pool is the same K/V with no
    # quantization -- the vendor kernel's own ground truth.
    fp8_pool = torch.cat((k_vals, v_vals), dim=-1).to(torch.float8_e4m3fn)
    bf16_pool = torch.cat((k_vals, v_vals), dim=-1).to(torch.bfloat16)
    # --- block tables: shared physical prefix + distinct page spans ---
    shared = [0, 1]
    rows = []
    for r in range(len(PREFILL_QLENS)):
        own = [2 + 10 * r + i for i in range(10)]  # 10 distinct pages each
        rows.append(shared + own)
    block_table = torch.tensor(rows, dtype=torch.int32)
    # --- queries: the real W1 q processing at each query's absolute
    # position (prefix + offset), as the model computes it ---
    total_q = sum(PREFILL_QLENS)
    q_pos = torch.tensor(
        [
            PREFILL_PREFIX[_req_index(PREFILL_QLENS, i)] + _offset_in(
                PREFILL_QLENS, i
            )
            for i in range(total_q)
        ],
        dtype=torch.int64,
    )
    q_cos = build_cos_sin_cache(int(q_pos.max()) + 1)[q_pos]
    xq = torch.randn((total_q, NQ, 128), generator=g) * 0.5
    q = (
        _W1P_VALUE_SCALE
        * ref.rope_partial_neox(
            ref.gemma_rmsnorm(xq, norm_ws["q"], EPS), q_cos
        )
    ).to(torch.bfloat16)
    # --- top-k: per (kv head, query), valid entries only below the
    # causal valid-block count, -1 sentinels above it ---
    cu = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(PREFILL_QLENS), 0)),
        dtype=torch.int32,
    )
    topk = torch.full((nkv, total_q, PREFILL_TOPK), -1, dtype=torch.int32)
    tg = torch.Generator().manual_seed(_W1P_NORM_W_SEED + 1)
    for i in range(total_q):
        r = _req_index(PREFILL_QLENS, i)
        q_abs = PREFILL_PREFIX[r] + _offset_in(PREFILL_QLENS, i)
        valid_blocks = (q_abs + BLOCK) // BLOCK
        real = min(PREFILL_TOPK, valid_blocks)
        if real <= 0:
            continue
        for kvh in range(nkv):
            ids = torch.randperm(valid_blocks, generator=tg)[:real]
            topk[kvh, i, :real] = ids.to(torch.int32)
    seq_lens = torch.tensor(
        [p + q for p, q in zip(PREFILL_PREFIX, PREFILL_QLENS)],
        dtype=torch.int32,
    )
    prefix_lens = torch.tensor(PREFILL_PREFIX, dtype=torch.int32)
    sm_scale = 1.0 / math.sqrt(128.0)
    return dict(
        q=q, pool=pool, fp8_pool=fp8_pool, bf16_pool=bf16_pool,
        index_cache=index_cache, k_vals=k_vals, v_vals=v_vals,
        block_table=block_table, topk=topk, cu_seqlens_q=cu,
        seq_lens=seq_lens, prefix_lens=prefix_lens, sm_scale=sm_scale,
        max_query_len=max(PREFILL_QLENS),
    )


def _req_index(qLens, i: int) -> int:
    pos = 0
    for r, length in enumerate(qLens):
        if pos <= i < pos + length:
            return r
        pos += length
    raise AssertionError(f"query {i} outside the chunk")


def _offset_in(qLens, i: int) -> int:
    pos = 0
    for length in qLens:
        if pos <= i < pos + length:
            return i - pos
        pos += length
    raise AssertionError(f"query {i} outside the chunk")


def validate_prefill_inputs(inp: dict) -> None:
    """Fail closed on any shape/range inconsistency in the case-5 inputs."""
    errors: list[str] = []
    total_q = sum(PREFILL_QLENS)
    q = inp["q"]
    if q.shape != (total_q, NQ, 128) or q.dtype != torch.bfloat16:
        errors.append(
            f"q must be [{total_q}, {NQ}, 128] bfloat16, got "
            f"{tuple(q.shape)} {q.dtype}"
        )
    pool = inp["pool"]
    if (
        pool.shape != (PREFILL_NUM_PAGES, 2 * NKV, BLOCK, 72)
        or pool.dtype != torch.uint8
    ):
        errors.append(
            f"pool must be [{PREFILL_NUM_PAGES}, {2 * NKV}, {BLOCK}, 72] "
            f"uint8, got {tuple(pool.shape)} {pool.dtype}"
        )
    elif (
        pool.stride(2) != 72 or pool.stride(1) != 9216 or pool.stride(3) != 1
    ):
        errors.append("pool slot regions must be contiguous")
    _validate_vendor_kv_pools(inp, PREFILL_NUM_PAGES, errors)
    ic = inp["index_cache"]
    if ic.shape != (PREFILL_NUM_PAGES * BLOCK, 128) or (
        ic.dtype != torch.float8_e4m3fn
    ):
        errors.append(
            "index cache must be the real fp8 e4m3 W1 index bytes "
            f"[{PREFILL_NUM_PAGES * BLOCK}, 128] float8_e4m3fn, got "
            f"{tuple(ic.shape)} {ic.dtype}"
        )
    bt = inp["block_table"]
    if bt.shape != (len(PREFILL_QLENS), PREFILL_MAX_BLOCKS) or not _integral(bt):
        errors.append(
            f"block_table must be [{len(PREFILL_QLENS)}, "
            f"{PREFILL_MAX_BLOCKS}] integral, got {tuple(bt.shape)} "
            f"{bt.dtype}"
        )
    else:
        lo, hi = int(bt.min()), int(bt.max())
        if lo < 0 or hi >= PREFILL_NUM_PAGES:
            errors.append(
                f"block_table entries must be physical pool pages in "
                f"[0, {PREFILL_NUM_PAGES}), got [{lo}, {hi}]"
            )
        if not torch.equal(
            bt[:, :2], bt[:1, :2].expand(bt.shape[0], 2)
        ):
            errors.append(
                "fixture block-table rows must share their physical prefix "
                "(the cross-slice page overlap the case exists for)"
            )
    cu = inp["cu_seqlens_q"]
    want_cu = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(PREFILL_QLENS), 0)),
        dtype=torch.int32,
    )
    if cu.shape != want_cu.shape or not _integral(cu) or (
        not torch.equal(cu, want_cu)
    ):
        errors.append(
            "cu_seqlens_q must be exactly the per-sequence query offsets of "
            f"{PREFILL_QLENS}"
        )
    seq = inp["seq_lens"]
    prefix = inp["prefix_lens"]
    want_seq = torch.tensor(
        [p + q for p, q in zip(PREFILL_PREFIX, PREFILL_QLENS)],
        dtype=torch.int32,
    )
    want_prefix = torch.tensor(PREFILL_PREFIX, dtype=torch.int32)
    if not torch.equal(seq, want_seq):
        errors.append(
            "seq_lens must be prefix + current query length per sequence "
            "(the real prefill invariant)"
        )
    if not torch.equal(prefix, want_prefix):
        errors.append(
            f"prefix_lens must be {PREFILL_PREFIX} (the real cached-context "
            "lengths the causal/valid-block math depends on)"
        )
    topk = inp["topk"]
    if topk.shape != (NKV, total_q, PREFILL_TOPK) or not _integral(topk):
        errors.append(
            f"topk must be [{NKV}, {total_q}, {PREFILL_TOPK}] integral, "
            f"got {tuple(topk.shape)} {topk.dtype}"
        )
    else:
        for i in range(total_q):
            r = _req_index(PREFILL_QLENS, i)
            q_abs = PREFILL_PREFIX[r] + _offset_in(PREFILL_QLENS, i)
            valid_blocks = (q_abs + BLOCK) // BLOCK
            real = min(PREFILL_TOPK, valid_blocks)
            for kvh in range(NKV):
                row = topk[kvh, i]
                below = row[:real]
                above = row[real:]
                if above.numel() and int(above.min()) != -1:
                    errors.append(
                        f"topk entries beyond the causal valid-block count "
                        f"of query {i} must be the -1 sentinel"
                    )
                    break
                if real and (
                    int(below.min()) < 0
                    or int(below.max()) >= valid_blocks
                    or int(below.unique().numel()) != real
                ):
                    errors.append(
                        f"topk entries of query {i} must be distinct valid "
                        f"logical blocks below valid_blocks={valid_blocks}"
                    )
                    break
    sm_scale = inp["sm_scale"]
    if not (sm_scale > 0 and math.isfinite(sm_scale)):
        errors.append(f"sm_scale must be positive and finite, got {sm_scale}")
    if errors:
        raise ValueError(
            "q4 KV harness prefill input data inconsistent (fix the harness "
            "data, not the kernel): " + "; ".join(errors)
        )
