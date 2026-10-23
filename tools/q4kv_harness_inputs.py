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
    fp8_pool = (
        torch.cat([k_vals, v_vals], dim=1).to(torch.float8_e4m3fn)
    )
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
        block_table=block_table, topk=topk, seq_lens=seq_lens,
        sm_scale=sm_scale,
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
    fp8_pool = inp["fp8_pool"]
    if fp8_pool.shape != (NUM_PAGES, 2 * NKV, BLOCK, 128) or (
        fp8_pool.dtype != torch.float8_e4m3fn
    ):
        errors.append(
            f"fp8 baseline pool must be [{NUM_PAGES}, {2 * NKV}, {BLOCK}, "
            f"128] float8_e4m3fn, got {tuple(fp8_pool.shape)} "
            f"{fp8_pool.dtype}"
        )
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
