"""Milestone-1 q4 (nvfp4-layout) KV cache tests (CPU-only; no vLLM, no GPU).

Covers:
- byte arithmetic of the pool (576 B/token/layer, 73728 B/page for 4 heads);
- scale-offset layout (K linear, V 4x4 token-quad swizzle), bijectivity;
- encoder oracles: e4m3 == torch's satfinite fp8 cast; e2m1 ladder;
- pack/unpack round trip + byte-exact repack + error bounds;
- full W1 reference (norm + rope + pack + scatter) vs independent math;
- fail-closed dispatch (validate_w1_call + overlay wrappers);
- pool detection (fp8/bf16/indexer/draft pools never match);
- staging dequant == per-slot oracle; prefill remap correctness;
- default-off: register() installs the overlay only under VLLM_EXL3_Q4KV=1;
- kernel boundary convention: fp8 e4m3 index buffers are handed to the
  Triton kernels as uint8 byte views (the GB10 val1 compile fix), the
  uint8 pool is passed as-is, and the GB10 harness frontend-checks the
  kernels before any GPU case;
- harness input consistency (the GB10 val2 follow-up): the reference
  raises a named ValueError (not an IndexError) when positions exceed
  the rotary cache, the shared case builders build a real partial-RoPE
  cache (theta 5e6) sized from positions, and the input validators
  fail closed on every out-of-range index before any kernel runs.

Triton is not importable on this VPS and there is no GPU: the Triton W1/R2
kernels are validated on GB10 by tools/gb10_q4kv_parity.py, not here.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

import vllm_exl3
import vllm_exl3.minimax_q4kv as q4kv
import vllm_exl3.minimax_q4kv_kernels as q4kernels
import vllm_exl3.minimax_q4kv_reference as ref

try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

NQ, NKV, NIQ = 8, 4, 4
ROTARY = 64
EPS = 1e-4


def _rand(rows: int, scale: float = 1.0, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn((rows, 128), generator=g) * scale).to(
        torch.bfloat16
    )


def _make_pool(num_pages: int, num_kv_heads: int = NKV) -> torch.Tensor:
    """A zeroed pool container with the bound nvfp4 geometry."""
    flat = torch.zeros(
        (num_pages, 2 * num_kv_heads, ref.SLOT_REGION), dtype=torch.uint8
    )
    # Emulate the bound 4-D container [nb, 2H, 128, 72] with contiguous
    # slot regions (stride(1) == 9216, stride(2) == 72).
    return flat.reshape(num_pages, 2 * num_kv_heads, 128, 72)


def _w1_kwargs(**over):
    kw = dict(
        qkv_dtype="bfloat16",
        qkv_width=(NQ + 2 * NKV + NIQ + 1) * 128,
        num_tokens=3,
        num_heads=NQ,
        num_kv_heads=NKV,
        num_index_heads=NIQ,
        rotary_dim=ROTARY,
        block_size=128,
        pool_dtype="uint8",
        pool_dim=4,
        pool_slots=2 * NKV,
        pool_tokens=128,
        pool_width=72,
        pool_row_stride=72,
        pool_slot_stride=9216,
        q_out_present=True,
        q_fp8_out_present=False,
        index_cache_dtype="float8_e4m3fn",
        index_q_out_dtype="float8_e4m3fn",
        kv_k_scale=1.0,
        kv_v_scale=1.0,
    )
    kw.update(over)
    return kw


# ---------------------------------------------------------------------------
# Pool arithmetic and layout
# ---------------------------------------------------------------------------


def test_pool_arithmetic() -> None:
    assert ref.nvfp4_row_bytes(128) == 72
    # page = 2 slots/head * 4 heads * 128 states * 72 B == 73728 B
    assert ref.q4_pool_page_bytes(NKV) == 2 * NKV * 128 * 72 == 73728
    # per-token per-layer = 576 B; over 60 layers = 33.75 KiB/token
    assert ref.q4_pool_bytes_per_token(NKV) == 576
    assert ref.q4_pool_bytes_per_token(NKV) * 60 == 576 * 60
    assert 576 * 60 == 34560  # 33.75 KiB
    # fp8 main KV was 1024 B/token/layer: q4 is 56.25% of it
    assert ref.q4_pool_bytes_per_token(NKV) / 1024 == 0.5625


def test_scale_offset_layout() -> None:
    # K (side 0) is linear: token*8 + group.
    for t in (0, 5, 127):
        for g in (0, 3, 7):
            assert ref.scale_block_offset(0, t, g) == t * 8 + g
    # V (side 1) uses the 4x4 token-quad swizzle.
    assert ref.scale_block_offset(1, 0, 0) == 0
    assert ref.scale_block_offset(1, 1, 0) == 1  # (t//4)*4+g//2=0, (g%2)*4+t%4=1
    assert ref.scale_block_offset(1, 0, 1) == 4
    assert ref.scale_block_offset(1, 7, 3) == (4 + 1) * 8 + 1 * 4 + 3
    # Both offset tables are bijective over the 1024-byte scale block.
    for side in (0, 1):
        table = ref._scale_offset_table(side, 128).reshape(-1)
        assert table.numel() == 1024
        assert torch.unique(table).numel() == 1024
        assert int(table.min()) == 0 and int(table.max()) == 1023


# ---------------------------------------------------------------------------
# Encoder oracles
# ---------------------------------------------------------------------------


def test_e4m3_encoder_matches_torch_cast() -> None:
    vals = torch.cat(
        [
            torch.linspace(-500, 500, 2001),
            torch.tensor(
                [0.0, 1e-12, 2**-9, 2**-9 * 1.5, 2**-6, 0.5, 447.9, 448.0,
                 448.5, 450.0, 1e30, float("inf"), float("-inf")]
            ),
        ]
    )
    ours = ref.e4m3_encode_satfinite(vals)
    torch_cast = vals.to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(ours, torch_cast)
    # NaN keeps the vendor NaN code with its sign bit.
    nan = ref.e4m3_encode_satfinite(torch.tensor([float("nan")]))
    assert nan[0] == 0x7F


def test_e2m1_ladder() -> None:
    values = ref.e2m1_code_values()
    assert values == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    mags = torch.tensor(list(values))
    codes = ref.e2m1_code(mags)
    assert codes.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    # decode inverts encode on the whole nibble space (sign incl.); e2m1
    # has no NaN -- code 7 is the saturation value 6 (vendor csrc).
    nibbles = torch.arange(16, dtype=torch.uint8)
    decoded = ref.e2m1_decode(nibbles)
    for n in range(16):
        mag = values[n & 7] if (n & 7) else 0.0
        want = -mag if n & 8 else mag
        assert float(decoded[n]) == want
    # The code is monotone in the magnitude between ladder points.
    assert int(ref.e2m1_code(torch.tensor([0.26]))) == 1
    assert int(ref.e2m1_code(torch.tensor([0.24]))) == 0


# ---------------------------------------------------------------------------
# Pack / unpack
# ---------------------------------------------------------------------------


def test_pack_unpack_roundtrip_and_repack() -> None:
    x = _rand(256, seed=1)
    data, scales = ref.pack_q4_row(x)
    assert data.shape == (256, 64) and scales.shape == (256, 8)
    back = ref.unpack_q4_row(data, scales)
    # 4-bit error bound: RNE to the ladder means |dq - x| is at most the
    # largest half-gap (1.0, between ladder points 4 and 6) times the
    # element's own group scale.
    scale = ref.e4m3_decode(scales).repeat_interleave(16, dim=-1)
    err = (back - x.to(torch.float32)).abs()
    assert bool((err <= scale * 1.0 + 1e-9).all())
    cos = torch.nn.functional.cosine_similarity(
        back, x.to(torch.float32), dim=-1
    )
    assert float(cos.min()) >= 0.99
    # Byte-exact repack: unpack(pack(x)) repacks to identical bytes.
    data2, scales2 = ref.pack_q4_row(back.to(torch.bfloat16))
    assert torch.equal(data, data2) and torch.equal(scales, scales2)


def test_slot_region_layout() -> None:
    rows = _rand(128, seed=2)
    for side in (0, 1):
        flat = ref.pack_slot_region_side(rows, side)
        assert flat.numel() == 9216
        # data block: token t's 64 bytes at t*64..t*64+64.
        packed_data, _ = ref.pack_q4_row(rows)
        assert torch.equal(
            flat[: 8192].view(128, 64), packed_data
        )
        # scales sit exactly at the side's offsets.
        _, packed_scales = ref.pack_q4_row(rows)
        off = ref._scale_offset_table(side, 128).reshape(-1)
        assert torch.equal(
            flat[8192:].gather(0, off).view(128, 8), packed_scales
        )
        # inverse: unpack returns the quantized values.
        vals = ref.unpack_slot_region_side(flat, side)
        again = ref.pack_slot_region_side(vals.to(torch.bfloat16), side)
        assert torch.equal(flat, again)


def test_pack_pool_page_roundtrip() -> None:
    rows = _rand(2 * NKV * 128, seed=3).view(2 * NKV, 128, 128)
    flat = ref.pack_pool_page(rows)
    assert flat.numel() == 2 * NKV * 9216
    vals = ref.unpack_pool_page(flat)
    again = ref.pack_pool_page(vals.to(torch.bfloat16))
    assert torch.equal(flat, again)


# ---------------------------------------------------------------------------
# W1 reference
# ---------------------------------------------------------------------------


def _w1_batch(num_tokens: int = 5, seed: int = 4):
    width = (NQ + 2 * NKV + NIQ + 1) * 128
    g = torch.Generator().manual_seed(seed)
    qkv = (torch.randn((num_tokens, width), generator=g) * 0.5).to(
        torch.bfloat16
    )
    q_norm_w = torch.randn(128, generator=g)
    k_norm_w = torch.randn(128, generator=g)
    iq_norm_w = torch.randn(128, generator=g)
    ik_norm_w = torch.randn(128, generator=g)
    max_pos = 64
    cos_sin = torch.randn((max_pos, ROTARY), generator=g)
    positions = torch.arange(num_tokens) + 3
    eps = 1e-4
    slot_mapping = torch.tensor([0, 1, 130, -1, 257], dtype=torch.int64)
    index_slot_mapping = torch.tensor([0, 1, 2, -1, 4], dtype=torch.int64)
    return (
        qkv, q_norm_w, k_norm_w, iq_norm_w, ik_norm_w, cos_sin, positions,
        eps, slot_mapping, index_slot_mapping,
    )


def test_reference_w1_write_layout() -> None:
    (
        qkv, q_norm_w, k_norm_w, iq_norm_w, ik_norm_w, cos_sin, positions,
        eps, slot_mapping, index_slot_mapping,
    ) = _w1_batch()
    out = ref.reference_w1_write(
        qkv, q_norm_w, k_norm_w, iq_norm_w, ik_norm_w, cos_sin, positions,
        eps, ROTARY, NQ, NKV, NIQ, slot_mapping, index_slot_mapping,
    )
    assert out["q"].shape == (5, NQ, 128)
    assert out["q"].dtype == torch.bfloat16
    assert out["index_q"].shape == (5, NIQ, 128)
    assert out["index_q"].dtype == torch.uint8
    assert out["pool"].shape == (3, 2 * NKV, 9216)  # sm max 257 -> page 2
    assert out["index_k"].shape == (5, 128)

    # Independent Q oracle: norm + rope per branch.
    x = qkv.to(torch.float32).view(5, -1, 128)
    cos_sin_f = cos_sin.to(torch.float32)[positions]
    qn = ref.rope_partial_neox(
        ref.gemma_rmsnorm(x[:, :NQ], q_norm_w, eps), cos_sin_f[:, :ROTARY]
    )
    assert torch.equal(out["q"], qn.to(torch.bfloat16))
    # index_q bytes: e4m3 of the bf16-quantized processed value.
    iqn = ref.rope_partial_neox(
        ref.gemma_rmsnorm(x[:, NQ + 2 * NKV: NQ + 2 * NKV + NIQ], iq_norm_w, eps),
        cos_sin_f[:, :ROTARY],
    )
    want = ref.e4m3_encode_satfinite(iqn.to(torch.bfloat16).to(torch.float32))
    assert torch.equal(out["index_q"], want)
    # index_k bytes: e4m3 of clamp(processed, +-448).
    ikn = ref.rope_partial_neox(
        ref.gemma_rmsnorm(x[:, -1], ik_norm_w, eps), cos_sin_f[:, :ROTARY]
    )
    assert torch.equal(out["index_k"], ref._e4m3_clamped(ikn))

    # Slot scatter: page/tok resolution matches slot_mapping.
    kn = ref.rope_partial_neox(
        ref.gemma_rmsnorm(
            x[:, NQ: NQ + NKV], k_norm_w, eps
        ), cos_sin_f[:, :ROTARY],
    )
    for i, sm in enumerate(slot_mapping.tolist()):
        if sm < 0:
            continue
        page, tok = sm // 128, sm % 128
        region_k = out["pool"][page, 0]
        region_v = out["pool"][page, 1]
        got_k = ref.unpack_slot_region_side(region_k, 0)[tok]
        got_v = ref.unpack_slot_region_side(region_v, 1)[tok]
        want_k = ref.unpack_q4_row(*ref.pack_q4_row(kn[i : i + 1]))[0]
        want_v = ref.unpack_q4_row(
            *ref.pack_q4_row(
                x[i : i + 1, NQ + NKV : NQ + NKV + 1].to(torch.bfloat16)
            )
        )[0]
        assert torch.equal(got_k, want_k[0])
        assert torch.equal(got_v, want_v[0])
    # Unmapped tokens leave the pool zero at their data positions.
    assert torch.equal(
        out["pool"][0, 0][3 * 64 : 3 * 64 + 8],
        torch.zeros(8, dtype=torch.uint8),
    )


def test_validate_w1_call_fail_closed() -> None:
    assert ref.validate_w1_call(**_w1_kwargs()) == []
    bad = [
        ("qkv_dtype", "float32"),
        ("qkv_width", 12345),
        ("num_heads", 0),
        ("num_index_heads", -1),
        ("rotary_dim", 130),
        ("block_size", 64),
        ("pool_dtype", "float8_e4m3fn"),
        ("pool_dim", 3),
        ("pool_slots", 2 * NKV - 2),
        ("pool_tokens", 64),
        ("pool_width", 256),
        ("pool_row_stride", 256),
        ("pool_slot_stride", 8192),
        ("q_out_present", False),
        ("q_fp8_out_present", True),
        ("index_cache_dtype", "bfloat16"),
        ("index_cache_dtype", "uint8"),  # nvfp4 indexer is milestone 2
        ("index_q_out_dtype", "bfloat16"),
        ("index_q_out_dtype", "uint8"),
        ("kv_k_scale", 2.0),
        ("kv_v_scale", 0.5),
    ]
    for key, value in bad:
        errors = ref.validate_w1_call(**_w1_kwargs(**{key: value}))
        assert errors, f"expected refusal for {key}={value!r}"
    # Refuse a pool that is not the nvfp4 container outright.
    assert ref.validate_w1_call(**_w1_kwargs(pool_dtype="nvfp4"))


# ---------------------------------------------------------------------------
# Overlay dispatch
# ---------------------------------------------------------------------------


def test_is_q4_pool_detection() -> None:
    pool = _make_pool(4)
    assert q4kv._is_q4_pool(pool)
    # fp8 main pool, bf16 pool, indexer pool, draft pool: never match.
    assert not q4kv._is_q4_pool(
        torch.zeros(4, NKV, 128, 256, dtype=torch.uint8)
    )
    assert not q4kv._is_q4_pool(
        torch.zeros(4, NKV, 128, 256, dtype=torch.bfloat16)
    )
    assert not q4kv._is_q4_pool(torch.zeros(4, 128, 128, dtype=torch.uint8))
    assert not q4kv._is_q4_pool(torch.zeros(4))
    assert not q4kv._is_q4_pool(None)
    # non-contiguous slot regions must not be treated as ours.
    wide = torch.zeros(4, 2 * NKV, 128, 72).transpose(2, 3)
    assert not q4kv._is_q4_pool(wide)


def _stub_kernels(monkeypatch, calls):
    mod = ModuleType("vllm_exl3.minimax_q4kv_kernels")

    def w1(*args, **kwargs):
        calls.append(("w1", args, kwargs))

    def decode(*args, **kwargs):
        calls.append(("decode", args, kwargs))

    mod.q4_w1_insert = w1
    mod.q4_sparse_decode = decode
    # The wrappers import this module lazily by name; patch both the
    # sys.modules entry and the package attribute so from-import resolves
    # to the stub.
    monkeypatch.setitem(sys.modules, "vllm_exl3.minimax_q4kv_kernels", mod)
    monkeypatch.setattr(
        vllm_exl3, "minimax_q4kv_kernels", mod, raising=False
    )


def _w1_call_args(pool):
    width = (NQ + 2 * NKV + NIQ + 1) * 128
    qkv = torch.zeros(3, width, dtype=torch.bfloat16)
    q_out = torch.zeros(3, NQ * 128, dtype=torch.bfloat16)
    # index_q is the e4m3 buffer the layer allocates (uint8 semantics).
    index_q_out = torch.zeros(3, NIQ * 128, dtype=torch.float8_e4m3fn)
    return dict(
        args=(
            qkv, torch.zeros(128), torch.zeros(128), torch.zeros(8, 64),
            torch.tensor([0, 1, 2]), NQ, NKV, ROTARY, EPS,
            torch.zeros(128), torch.zeros(128), NIQ,
            torch.tensor([0, 1, -1]), torch.tensor([0, 1, 2]),
            pool, torch.zeros(4, 128, 128, dtype=torch.uint8), 128,
            q_out, index_q_out, "nvfp4",
        ),
        kwargs=dict(
            q_fp8_out=None,
            q_fp8_scale=1.0,
            kv_k_scale=torch.tensor(1.0),
            kv_v_scale=None,
            write_qkv=True,
        ),
    )


def test_w1_wrapper_passes_fp8_to_original(monkeypatch) -> None:
    calls = []
    orig = lambda *a, **k: calls.append(("orig", a, k))  # noqa: E731
    wrapper = q4kv._make_w1_wrapper(orig)
    fp8_pool = torch.zeros(4, NKV, 128, 256, dtype=torch.uint8)
    spec = _w1_call_args(fp8_pool)
    wrapper(*spec["args"], **spec["kwargs"])
    assert calls[0][0] == "orig"
    assert calls[0][1] == spec["args"]


def test_w1_wrapper_q4_fail_closed(monkeypatch) -> None:
    calls = []
    _stub_kernels(monkeypatch, calls)
    orig = lambda *a, **k: calls.append(("orig", a, k))  # noqa: E731
    wrapper = q4kv._make_w1_wrapper(orig)
    pool = _make_pool(4)
    base = _w1_call_args(pool)

    def run(pos_over=None, kw_over=None):
        args = list(base["args"])
        kw = dict(base["kwargs"])
        if pos_over:
            for pos, value in pos_over.items():
                args[pos] = value
        kw.update(kw_over or {})
        wrapper(*args, **kw)

    # non-nvfp4 dtype label on a q4 pool -> refuse
    with pytest.raises(ValueError):
        run(pos_over={19: "fp8"})
    # fp8 query -> refuse
    with pytest.raises(ValueError):
        run(kw_over={"q_fp8_out": torch.zeros(3, NQ * 128, dtype=torch.uint8)})
    # write_qkv=False -> refuse
    with pytest.raises(ValueError):
        run(kw_over={"write_qkv": False})
    # missing bf16 q_out -> refuse
    with pytest.raises(ValueError):
        run(pos_over={17: None})
    # non-unit scales -> refuse
    with pytest.raises(ValueError):
        run(kw_over={"kv_k_scale": torch.tensor(2.0)})
    with pytest.raises(ValueError):
        run(kw_over={"kv_v_scale": 0.5})
    # slot-count mismatch -> refuse
    with pytest.raises(ValueError):
        run(pos_over={6: NKV + 2})
    # skip_index_branch with index outputs -> refuse
    with pytest.raises(ValueError):
        run(kw_over={"skip_index_branch": True})
    # unknown kwarg -> TypeError
    with pytest.raises(TypeError):
        run(kw_over={"nope": 1})
    # nothing reached the kernels or the original op in any refusal above
    assert calls == []


def test_w1_wrapper_q4_routes_to_w1_and_skip_index(monkeypatch) -> None:
    calls = []
    _stub_kernels(monkeypatch, calls)
    orig = lambda *a, **k: calls.append(("orig", a, k))  # noqa: E731
    wrapper = q4kv._make_w1_wrapper(orig)
    pool = _make_pool(4)
    base = _w1_call_args(pool)
    wrapper(*base["args"], **base["kwargs"])
    assert calls[0][0] == "w1"
    # index branch is served (num_index_heads preserved, index outputs
    # passed); q4_w1_insert's positional order: ..., num_index_heads=11,
    # ..., index_cache=15, ..., index_q_out=18.
    w1_args = calls[0][1]
    assert w1_args[11] == NIQ
    assert w1_args[15] is not None
    assert w1_args[18] is not None
    # skip_index_branch strips the index surface entirely (the index-topk
    # reuse layers pass no index buffers).
    calls.clear()
    args = list(base["args"])
    args[15] = None  # index_cache
    args[18] = None  # index_q_out
    wrapper(*args, skip_index_branch=True)
    assert calls[0][0] == "w1"
    assert calls[0][1][11] == 0
    assert calls[0][1][15] is None
    assert calls[0][1][18] is None


def _decode_args(pool):
    q = torch.zeros(4, NQ, 128, dtype=torch.bfloat16)
    topk = torch.zeros(NKV, 4, 8, dtype=torch.int32)
    bt = torch.zeros(2, 4, dtype=torch.int32)
    seq = torch.tensor([16, 16], dtype=torch.int32)
    out = torch.zeros(4, NQ, 128, dtype=torch.bfloat16)
    return (q, pool, topk, bt, seq, NKV, 0.08, out, 2)


def test_decode_wrapper_dispatch(monkeypatch) -> None:
    calls = []
    _stub_kernels(monkeypatch, calls)
    orig = lambda *a, **k: calls.append(("orig", a, k))  # noqa: E731
    wrapper = q4kv._make_decode_wrapper(orig)
    # fp8 pool -> original
    wrapper(*_decode_args(torch.zeros(4, NKV, 128, 256, dtype=torch.uint8)))
    assert calls[0][0] == "orig"
    # q4 pool, unit scales -> R2
    wrapper(*_decode_args(_make_pool(4)), k_scale=None, v_scale=None)
    assert calls[1][0] == "decode"
    # q4 pool, non-unit scale -> refuse
    with pytest.raises(ValueError):
        wrapper(*_decode_args(_make_pool(4)), k_scale=torch.tensor(2.0))
    assert len(calls) == 2


def test_prefill_wrapper_stages_and_remaps(monkeypatch) -> None:
    calls = []
    orig = lambda *a, **k: calls.append(("orig", a, k))  # noqa: E731
    wrapper = q4kv._make_prefill_wrapper(orig)
    # fp8 pool -> original, untouched
    wrapper(*_prefill_args(torch.zeros(4, NKV, 128, 256, dtype=torch.uint8)))
    assert calls[0][0] == "orig"
    assert calls[0][1][1].shape[-1] == 256

    # q4 pool -> staged bf16 pool + remapped block table.
    pool_flat, bt, topk, k_pages, v_pages = _staged_fixture()
    pool = pool_flat.reshape(4, 2 * NKV, 128, 72)
    spec = _prefill_args(pool, bt=bt, topk=topk)
    wrapper(*spec)
    assert calls[1][0] == "orig"
    staged = calls[1][1][1]
    fake_bt = calls[1][1][3]
    assert staged.dtype == torch.bfloat16
    assert staged.shape == (3, NKV, 128, 256)  # 2 selected pages + zero page
    # The staged pages hold exactly the dequantized selected pages' K|V.
    sel = [1, 2]  # physical pages selected by the fixture
    for row, page in enumerate(sel):
        assert torch.equal(
            staged[row, :, :, :128], k_pages[page].to(torch.bfloat16)
        )
        assert torch.equal(
            staged[row, :, :, 128:], v_pages[page].to(torch.bfloat16)
        )
    # zero page is zero; every unselected physical page maps to it and
    # every selected one maps to its staged row.
    assert torch.equal(staged[2], torch.zeros_like(staged[2]))
    for r in range(bt.shape[0]):
        for b in range(bt.shape[1]):
            phys = int(bt[r, b])
            if phys in sel:
                assert int(fake_bt[r, b]) == sel.index(phys)
            else:
                assert int(fake_bt[r, b]) == 2
    # the non-pool call arguments are passed through unchanged; only the
    # pool (1) and block table (3) are replaced by the staged pair.
    got = calls[1][1]
    # tensors are the very same objects
    for i in (0, 2, 4, 5, 6, 7, 10):
        assert got[i] is spec[i]
    # plain ints pass through with the same value
    assert got[8] == NKV and got[9] == spec[9]


def _prefill_args(pool, bt=None, topk=None):
    q = torch.zeros(4, NQ, 128, dtype=torch.bfloat16)
    if bt is None:
        bt = torch.zeros(2, 4, dtype=torch.int32)
    if topk is None:
        topk = torch.zeros(NKV, 4, 8, dtype=torch.int32)
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    seq = torch.tensor([16, 16], dtype=torch.int32)
    ctx = torch.tensor([0, 0], dtype=torch.int32)
    out = torch.zeros(4, NQ, 128, dtype=torch.bfloat16)
    return (q, pool, topk, bt, cu, seq, ctx, 16, NKV, 0.08, out)


def _staged_fixture():
    """Pack random K/V pages into a pool; select pages 1 and 2."""
    flat = torch.zeros((4, 2 * NKV, ref.SLOT_REGION), dtype=torch.uint8)
    for page in range(4):
        rows = _rand(2 * NKV * 128, seed=100 + page)  # distinct per page
        for slot in range(2 * NKV):
            flat[page, slot] = ref.pack_slot_region_side(
                rows[slot * 128 : (slot + 1) * 128], slot % 2
            )
    values = ref.unpack_pool_page(flat.reshape(-1))
    k_pages = values.view(4, 2 * NKV, 128, 128)[:, 0::2]
    v_pages = values.view(4, 2 * NKV, 128, 128)[:, 1::2]
    assert not torch.equal(k_pages[0], k_pages[1])
    bt = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.int32)
    topk = torch.full((NKV, 4, 8), -1, dtype=torch.int32)  # sentinels
    topk[:, :, 0] = 1  # logical block 1 selected everywhere
    topk[0, 0, 1] = 2  # plus logical block 2 for one query
    return flat, bt, topk, k_pages, v_pages


def test_dequant_pool_pages_matches_oracle() -> None:
    flat, _, _, _, _ = _staged_fixture()
    got = ref.dequant_pool_pages(flat)
    oracle = torch.zeros_like(got)
    for page in range(flat.shape[0]):
        for slot in range(flat.shape[1]):
            oracle[page, slot] = ref.unpack_slot_region_side(
                flat[page, slot], slot % 2
            )
    assert torch.equal(got, oracle)


def test_build_prefill_staging_all_selected(monkeypatch) -> None:
    flat, bt, topk, k_pages, v_pages = _staged_fixture()
    pool = flat.reshape(4, 2 * NKV, 128, 72)
    staging, fake_bt = ref.build_prefill_staging(pool, bt, topk, NKV)
    assert staging.shape == (3, NKV, 128, 256)
    assert torch.equal(staging[0, :, :, :128], k_pages[1].to(torch.bfloat16))
    assert torch.equal(staging[1, :, :, :128], k_pages[2].to(torch.bfloat16))
    assert torch.equal(staging[0, :, :, 128:], v_pages[1].to(torch.bfloat16))
    assert torch.equal(staging[1, :, :, 128:], v_pages[2].to(torch.bfloat16))
    assert torch.equal(staging[2], torch.zeros_like(staging[2]))
    # nothing selected -> only the zero page, everything remaps to it
    empty_topk = torch.full((NKV, 4, 8), -1, dtype=torch.int32)
    staging2, fake_bt2 = ref.build_prefill_staging(pool, bt, empty_topk, NKV)
    assert staging2.shape == (1, NKV, 128, 256)
    assert torch.equal(fake_bt2, torch.full_like(bt, 0))


# ---------------------------------------------------------------------------
# Default-off gate
# ---------------------------------------------------------------------------


def _stub_all_installers(monkeypatch, calls):
    names = [
        ("vllm_exl3.bf16_madv_compat", "install_mixed_bf16_madv_compat"),
        ("vllm_exl3.deepseek_v41", "install_deepseek_v41_compat"),
        ("vllm_exl3.k78_compat", "install_k78_config_compat"),
        ("vllm_exl3.mixed_k_guard", "install_mixed_k_prescan_guard"),
        ("vllm_exl3.physical_k_compat", "install_physical_fused_k_compat"),
        ("vllm_exl3.runtime_policy", "install_native_row_policy"),
        ("vllm_exl3.tp_geometry_compat", "install_tp_geometry_compat"),
        ("vllm_exl3.uva_offload", "install_uva_expert_validation"),
    ]
    fake_exl3 = ModuleType("vllm_exl3.exl3")
    monkeypatch.setitem(sys.modules, "vllm_exl3.exl3", fake_exl3)
    monkeypatch.setattr(vllm_exl3, "exl3", fake_exl3, raising=False)
    for module_name, function_name in names:
        mod = ModuleType(module_name)

        def installer(*a, _n=function_name, **k):
            calls.append(_n)

        setattr(mod, function_name, installer)
        monkeypatch.setitem(sys.modules, module_name, mod)


def test_register_default_off(monkeypatch) -> None:
    calls: list = []
    _stub_all_installers(monkeypatch, calls)
    monkeypatch.delenv("VLLM_EXL3_Q4KV", raising=False)
    vllm_exl3.register()
    assert calls.count("q4kv_install") == 0
    assert not any("q4" in c for c in calls)


def test_register_opt_in_installs_overlay(monkeypatch) -> None:
    calls: list = []
    _stub_all_installers(monkeypatch, calls)
    q4mod = ModuleType("vllm_exl3.minimax_q4kv")

    def install():
        calls.append("q4kv_install")

    q4mod.install = install
    monkeypatch.setitem(sys.modules, "vllm_exl3.minimax_q4kv", q4mod)
    monkeypatch.setenv("VLLM_EXL3_Q4KV", "1")
    vllm_exl3.register()
    assert calls.count("q4kv_install") == 1


def test_triton_availability_documented() -> None:
    """Triton is absent on this VPS: the W1/R2 kernels are GB10-only."""
    if HAS_TRITON:
        pytest.skip("triton importable; interpreter tests belong upstream")
    with pytest.raises(ImportError):
        import triton  # noqa: F401


def test_kernels_module_needs_triton_at_call_time() -> None:
    """The kernel wrappers import triton only when invoked (CPU-safe import)."""
    assert not hasattr(q4kernels, "_W1_KERNEL") or q4kernels._W1_KERNEL is None
    if not HAS_TRITON:
        with pytest.raises(ImportError):
            q4kernels._kernels()


# ---------------------------------------------------------------------------
# Kernel boundary convention: fp8 buffers cross as uint8 byte views
# ---------------------------------------------------------------------------


class _RecordingKernel:
    """Stands in for a @triton.jit kernel and records every launch."""

    def __init__(self, calls: list, tag: str) -> None:
        self._calls = calls
        self._tag = tag

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self._calls.append((self._tag, grid, args, kwargs))

        return launch


def _install_kernel_recorders(monkeypatch, calls) -> None:
    """Patch the built-kernel cache so the real wrappers launch recorders."""
    monkeypatch.setattr(
        q4kernels,
        "_W1_KERNEL",
        _RecordingKernel(calls, "w1"),
        raising=False,
    )
    monkeypatch.setattr(
        q4kernels,
        "_DECODE_KERNEL",
        _RecordingKernel(calls, "decode"),
        raising=False,
    )


def _real_w1_inputs():
    """A validate-clean CPU W1 call set (mirrors the GB10 harness case 1)."""
    num_tokens = 3
    width = (NQ + 2 * NKV + NIQ + 1) * 128
    qkv = torch.zeros(num_tokens, width, dtype=torch.bfloat16)
    q_out = torch.zeros(num_tokens, NQ, 128, dtype=torch.bfloat16)
    index_q_out = torch.zeros(num_tokens, NIQ, 128,
                              dtype=torch.float8_e4m3fn)
    index_cache = torch.zeros(4, 128, 128, dtype=torch.float8_e4m3fn)
    pool = _make_pool(4)
    norm_w = torch.zeros(128)
    cos_sin = torch.zeros(256, ROTARY)
    positions = torch.tensor([1000, 1001, 1002])
    slot_mapping = torch.tensor([128, 129, -1])
    index_slot_mapping = torch.tensor([0, 1, -1])
    return dict(
        qkv=qkv, q_out=q_out, index_q_out=index_q_out,
        index_cache=index_cache, pool=pool, norm_w=norm_w, cos_sin=cos_sin,
        positions=positions, slot_mapping=slot_mapping,
        index_slot_mapping=index_slot_mapping,
    )


def test_w1_insert_hands_fp8_buffers_as_uint8_views(monkeypatch) -> None:
    """GB10 val1 fix: the fp8 index buffers must reach the kernel as uint8
    byte views of the same storage; a raw fp8 pointer makes Triton's
    frontend try to cast the stored bytes to fp8e4nv and fail to compile.
    """
    calls: list = []
    _install_kernel_recorders(monkeypatch, calls)
    inp = _real_w1_inputs()
    q4kernels.q4_w1_insert(
        inp["qkv"], inp["norm_w"], inp["norm_w"], inp["cos_sin"],
        inp["positions"], NQ, NKV, ROTARY, EPS,
        inp["norm_w"], inp["norm_w"], NIQ,
        inp["slot_mapping"], inp["index_slot_mapping"], inp["pool"],
        inp["index_cache"], 128, inp["q_out"], inp["index_q_out"],
    )
    assert len(calls) == 1
    tag, grid, args, kwargs = calls[0]
    assert tag == "w1"
    assert grid == (3 * (NQ + 2 * NKV + NIQ + 1),)
    assert kwargs["PROCESS_INDEX"] is True
    # launch order: ..., 2=index_q_out view, 11=pool, 12=index_cache view.
    assert args[0] is inp["qkv"]
    assert args[1] is inp["q_out"]
    assert args[11] is inp["pool"]
    assert args[8] is inp["positions"]
    for view, fp8 in ((args[2], inp["index_q_out"]),
                     (args[12], inp["index_cache"])):
        assert view.dtype == torch.uint8
        assert view.data_ptr() == fp8.data_ptr()
        assert view is not fp8
    # No fp8-typed tensor crosses the boundary at all.
    for a in args:
        if isinstance(a, torch.Tensor):
            assert a.dtype != torch.float8_e4m3fn


def test_w1_insert_none_fallbacks_are_not_viewed(monkeypatch) -> None:
    """Without index buffers the wrapper passes the documented in-place
    placeholders (qkv / pool); the placeholder is never dtype-viewed.
    """
    calls: list = []
    _install_kernel_recorders(monkeypatch, calls)
    inp = _real_w1_inputs()
    q4kernels.q4_w1_insert(
        inp["qkv"], inp["norm_w"], inp["norm_w"], inp["cos_sin"],
        inp["positions"], NQ, NKV, ROTARY, EPS,
        None, None, NIQ,
        inp["slot_mapping"], inp["index_slot_mapping"], inp["pool"],
        None, 128, inp["q_out"], None,
    )
    assert len(calls) == 1
    args = calls[0][2]
    assert args[2] is inp["qkv"]
    assert args[12] is inp["pool"]


def test_decode_wrapper_boundary_convention(monkeypatch) -> None:
    """R2 takes no fp8 tensors: the uint8 pool must be handed over
    as-is (no view, no copy) and the split-K partials must be bf16/fp32.
    The unchanged vLLM merge kernel is reached through a stub module.
    """
    calls: list = []
    _install_kernel_recorders(monkeypatch, calls)
    merge_calls: list = []

    class _MergeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                merge_calls.append((grid, args, kwargs))

            return launch

    leaf_name = "vllm.models.minimax_m3.common.ops.sparse_attn"
    parents = ["vllm", "vllm.models", "vllm.models.minimax_m3",
               "vllm.models.minimax_m3.common",
               "vllm.models.minimax_m3.common.ops"]
    for name in parents:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    leaf = ModuleType(leaf_name)
    setattr(leaf, "_merge_topk_attn_out_kernel", _MergeKernel())
    monkeypatch.setitem(sys.modules, leaf_name, leaf)

    q = torch.zeros(4, NQ, 128, dtype=torch.bfloat16)
    pool = _make_pool(4)
    topk = torch.zeros(NKV, 4, 8, dtype=torch.int32)
    block_table = torch.zeros(4, 8, dtype=torch.int32)
    seq_lens = torch.full((4,), 8 * 128, dtype=torch.int32)
    out = torch.zeros_like(q)
    q4kernels.q4_sparse_decode(
        q, pool, topk, block_table, seq_lens, NKV, 0.08, out, 1
    )
    assert [c[0] for c in calls] == ["decode"]
    _, grid, args, kwargs = calls[0]
    assert args[0] is q
    assert args[1] is pool  # uint8 pool, unchanged identity
    assert args[1].dtype == torch.uint8
    o_partial, lse_partial = args[3], args[4]
    assert o_partial.dtype == torch.bfloat16
    assert o_partial.shape == (kwargs["NUM_TOPK_CHUNKS"], 4, NQ, 128)
    assert lse_partial.dtype == torch.float32
    assert lse_partial.shape == (kwargs["NUM_TOPK_CHUNKS"], 4, NQ)
    for a in args:
        if isinstance(a, torch.Tensor):
            assert a.dtype != torch.float8_e4m3fn
    # The unchanged vLLM merge kernel consumes exactly those partials.
    assert len(merge_calls) == 1
    m_args, m_kwargs = merge_calls[0][1], merge_calls[0][2]
    assert m_args[0] is o_partial and m_args[1] is lse_partial
    assert m_args[2] is out
    assert m_kwargs["NUM_TOPK_CHUNKS"] == kwargs["NUM_TOPK_CHUNKS"]


def test_gb10_harness_frontend_checks_before_any_gpu_case() -> None:
    """The GB10 harness must run the sm_121 frontend compile check before
    any GPU unit (fail fast with a clear message), and the check tool must
    pin the sm_121 target and the uint8-view convention."""
    root = Path(__file__).resolve().parents[1]
    harness = (root / "tools" / "gb10_q4kv_parity.py").read_text()
    check = (root / "tools" / "q4kv_triton_frontend_check.py").read_text()
    units = harness[harness.index("    units = [") :]
    gate = units.index('("0/frontend-check", _frontend_check_unit)')
    first_gpu = units.index('("1a/w1-norm-rope-numerics", unit_1a)')
    assert gate < first_gpu, "frontend check must run before the W1 units"
    assert 'GPUTarget("cuda", 121, 32)' in check
    assert '"index_q_out_ptr": "*u8"' in check
    assert '"index_cache_ptr": "*u8"' in check


# ---------------------------------------------------------------------------
# Harness input consistency (GB10 val2 follow-up): the reference position
# guard and the shared builders/validators in tools/q4kv_harness_inputs.py
# ---------------------------------------------------------------------------

_HARNESS_INPUTS = None


def _harness_inputs_module():
    """Load tools/q4kv_harness_inputs.py (pure torch; no triton needed)."""
    global _HARNESS_INPUTS
    if _HARNESS_INPUTS is None:
        import importlib.util

        root = Path(__file__).resolve().parents[1]
        path = root / "tools" / "q4kv_harness_inputs.py"
        spec = importlib.util.spec_from_file_location(
            "q4kv_harness_inputs", path
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["q4kv_harness_inputs"] = mod
        spec.loader.exec_module(mod)
        _HARNESS_INPUTS = mod
    return _HARNESS_INPUTS


def test_reference_w1_write_positions_beyond_cache_fail_closed() -> None:
    """The GB10 val2 crash class: positions beyond the rotary cache must
    raise a ValueError that names the input, never an IndexError that
    looks like a kernel bug. In-range positions must keep working."""
    (
        qkv, q_norm_w, k_norm_w, iq_norm_w, ik_norm_w, cos_sin, positions,
        eps, slot_mapping, index_slot_mapping,
    ) = _w1_batch()
    rows = cos_sin.shape[0]
    for bad in (
        positions + rows,  # one past the last cache row
        torch.full_like(positions, rows + 1000),  # the val2 distances
    ):
        with pytest.raises(ValueError, match="cos_sin"):
            ref.reference_w1_write(
                qkv, q_norm_w, k_norm_w, iq_norm_w, ik_norm_w, cos_sin,
                bad, eps, ROTARY, NQ, NKV, NIQ, slot_mapping,
                index_slot_mapping,
            )
    with pytest.raises(ValueError, match="positions"):
        ref.reference_w1_write(
            qkv, q_norm_w, k_norm_w, iq_norm_w, ik_norm_w, cos_sin,
            positions - int(positions.max()) - 1, eps, ROTARY, NQ, NKV,
            NIQ, slot_mapping, index_slot_mapping,
        )
    # Regression guard: the in-range batch still packs.
    out = ref.reference_w1_write(
        qkv, q_norm_w, k_norm_w, iq_norm_w, ik_norm_w, cos_sin,
        positions, eps, ROTARY, NQ, NKV, NIQ, slot_mapping,
        index_slot_mapping,
    )
    assert out["q"].shape[0] == positions.shape[0]


def test_harness_inputs_are_consistent_and_validators_accept_them() -> None:
    """The shipped case builders must pass their own validators (the CPU
    dry run and the GB10 harness consume the very same bytes)."""
    hi = _harness_inputs_module()
    hi.validate_w1_inputs(hi.w1_inputs())
    hi.validate_decode_inputs(hi.decode_inputs())


def test_harness_rotary_cache_is_the_real_partial_rope() -> None:
    """The rotary cache must be the real MiniMax-M3 partial-RoPE table
    (theta 5e6, rotary 64, vLLM's NeoX halved row form), sized to cover
    max(positions) + 1 — not the val2 harness's 256 random rows."""
    import math

    hi = _harness_inputs_module()
    assert hi.ROPE_THETA == 5_000_000.0
    assert hi.ROTARY == 64
    inp = hi.w1_inputs()
    rows, width = inp["cos_sin"].shape
    pos_max = int(inp["positions"].max())
    assert (rows, width) == (pos_max + 1, hi.ROTARY)
    inv_freq = 1.0 / (
        hi.ROPE_THETA
        ** (
            torch.arange(0, hi.ROTARY, 2, dtype=torch.float32) / hi.ROTARY
        )
    )
    p = torch.tensor([7, 1000, pos_max], dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", p, inv_freq)
    want = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    assert torch.allclose(inp["cos_sin"][[7, 1000, pos_max]], want)
    assert torch.allclose(
        inp["cos_sin"][:64], hi.build_cos_sin_cache(64)
    )
    assert abs(float(inp["cos_sin"][1000, 0]) - math.cos(1000.0)) < 1e-5


def test_harness_validator_catches_the_val2_position_bug() -> None:
    """The exact GB10 val2 data bug (positions 1000.. with a 256-row
    cache) must be caught by the validator before anything runs."""
    hi = _harness_inputs_module()
    inp = hi.w1_inputs()
    inp["cos_sin"] = torch.randn(
        256, hi.ROTARY, generator=torch.Generator().manual_seed(0)
    )
    with pytest.raises(ValueError, match="cos_sin cache"):
        hi.validate_w1_inputs(inp)


def test_harness_validator_catches_index_range_bugs() -> None:
    """Every index of every generated input must be in range for its
    tensor; each mutation below is a silent OOB write/read in the
    kernel and must fail closed in the validator."""
    hi = _harness_inputs_module()
    cases = [
        ("slot_mapping beyond the pool", "slot_mapping",
         lambda d, k: d[k].__setitem__(-1, hi.NUM_PAGES * hi.BLOCK),
         "slot_mapping"),
        ("index_slot_mapping beyond the index cache",
         "index_slot_mapping",
         lambda d, k: d[k].__setitem__(-1, 2 * hi.BLOCK),
         "index_slot_mapping"),
        ("index_cache None (pool placeholder corruption)", "index_cache",
         lambda d, k: d.update(index_cache=None), "index_cache"),
        ("positions beyond the cache", "positions",
         lambda d, k: d[k].__setitem__(
             0, int(d["cos_sin"].shape[0]) + 5
         ), "cos_sin cache"),
    ]
    for name, key, mutate, needle in cases:
        bad = hi.w1_inputs()
        mutate(bad, key)
        with pytest.raises(ValueError, match=needle):
            hi.validate_w1_inputs(bad)

    dec = hi.decode_inputs()
    dcases = [
        ("topk beyond the block table width", lambda d: d["topk"].__setitem__(
            (0, 0, -1), d["block_table"].shape[1]), "topk"),
        ("block_table beyond the pool pages",
         lambda d: d["block_table"].__setitem__((0, -1), hi.NUM_PAGES),
         "block_table"),
        ("seq_lens beyond the block table capacity",
         lambda d: d["seq_lens"].fill_(hi.NUM_PAGES * hi.BLOCK + 128),
         "seq_lens"),
        ("seq_lens off the fixture invariant",
         lambda d: d["seq_lens"].fill_(hi.TOPK * hi.BLOCK + 128),
         "fixture seq_lens"),
    ]
    for name, mutate, needle in dcases:
        bad = hi.decode_inputs()
        mutate(bad)
        with pytest.raises(ValueError, match=needle):
            hi.validate_decode_inputs(bad)


def test_gb10_harness_validates_inputs_before_every_case() -> None:
    """Every harness case must run its input validator before any kernel
    call, and the CPU dry run must run the same validators (a data bug
    must never look like a kernel bug)."""
    root = Path(__file__).resolve().parents[1]
    harness = (root / "tools" / "gb10_q4kv_parity.py").read_text()

    def body(marker: str) -> str:
        start = harness.index(marker)
        nxt = harness.index("\ndef ", start + 1)
        return harness[start:nxt]

    # The shared W1 evidence validates before the Triton launch, and
    # every 1x unit consumes that evidence.
    w1 = body("class _W1Evidence")
    assert w1.index("hi.validate_w1_inputs") < w1.index("q4k.q4_w1_insert")
    for unit in ("def unit_1a", "def unit_1b", "def unit_1c",
                 "def unit_1d"):
        assert "_w1_evidence()" in body(unit), (
            f"{unit} must gate the shared W1 evidence"
        )
    # The decode fixture validates before any R2 launch (units 2 and 4
    # reach the kernel only through _run_r2, which the fixture precedes).
    dec = body("def _decode_fixture")
    assert dec.index("hi.validate_decode_inputs") < dec.index("return inp")
    for unit in ("def unit_2", "def unit_4"):
        b = body(unit)
        assert b.index("_decode_fixture()") < b.index("_run_r2")
    run_r2 = body("def _run_r2")
    assert run_r2.index("q4k.q4_sparse_decode") > 0
    # The dry-run reference units validate their inputs too.
    for unit, act in (
        ("def _dry_w1_reference", "ref.reference_w1_write"),
        ("def _dry_gate_a", "ref.gemma_rmsnorm"),
        ("def _dry_gate_b", "ref.reference_w1_write"),
        ("def _dry_decode", "ref.reference_sparse_attention"),
    ):
        b = body(unit)
        assert b.index("hi.validate_") < b.index(act), (
            f"{unit} must validate inputs before {act}"
        )
    main_body = harness[harness.index("def main") :]
    assert '"--cpu-dry-run"' in main_body and "cpu_dry_run()" in main_body


def test_gb10_harness_runs_every_unit_to_completion() -> None:
    """The GB10 val3 lesson: one run must give the full picture. The
    harness must (a) wrap every unit so an exception is recorded as that
    unit's FAIL instead of aborting the run, (b) print one PASS/FAIL
    line per unit, (c) write the full diagnostics as a JSON receipt when
    --json is given, and (d) exit non-zero when any unit failed."""
    root = Path(__file__).resolve().parents[1]
    harness = (root / "tools" / "gb10_q4kv_parity.py").read_text()
    runner = harness[harness.index("def run_units") :]
    assert "except Exception" in runner, "per-unit exception capture"
    assert 'entry = {"case": name, "status"' in runner
    assert "json.dumps(receipt" in runner
    assert 'return 0 if verdict == "PASS" else 1' in runner
    units = harness[harness.index("    units = [") :]
    for name in (
        "0/frontend-check",
        "1a/w1-norm-rope-numerics",
        "1b/w1-packing-bytes",
        "1c/w1-index-bytes",
        "1d/w1-rewrite-scope",
        "2/r2-decode-accuracy",
        "3/pool-bytes",
        "4/cuda-graph-replay",
    ):
        assert f'("{name}"' in units, f"missing harness unit {name}"


# ---------------------------------------------------------------------------
# GB10 val3 follow-up: the parity gates (tools/q4kv_gates.py), the rz
# payload-quotient primitive, and the kernel's mirror of it
# ---------------------------------------------------------------------------

_Q4KV_GATES = None


def _gates_module():
    """Load tools/q4kv_gates.py (pure torch; no triton needed)."""
    global _Q4KV_GATES
    if _Q4KV_GATES is None:
        import importlib.util

        root = Path(__file__).resolve().parents[1]
        path = root / "tools" / "q4kv_gates.py"
        spec = importlib.util.spec_from_file_location("q4kv_gates", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["q4kv_gates"] = mod
        spec.loader.exec_module(mod)
        _Q4KV_GATES = mod
    return _Q4KV_GATES


def test_gate_bf16_numerics_semantics() -> None:
    """Gate (a): sparse 1-ulp noise (the expected kernel-vs-reference
    shape) passes; a uniform 1-ulp shift (systematic bias), gross
    corruption, sign flips and one-sided NaN fail; the reported metrics
    are exact."""
    g = _gates_module()
    gen = torch.Generator().manual_seed(0)
    base = torch.randn(64, 128, generator=gen).to(torch.bfloat16)
    # identical rows: pass with zero damage.
    r = g.gate_bf16_numerics(base, base, "t")
    assert r["passed"] and r["max_ulp"] == 0 and r["n_mismatch"] == 0
    assert abs(r["cosine"] - 1.0) < 1e-9
    # sparse 1-ulp perturbation: must PASS (this is the whole point of
    # the val3 rework — the reference's precise 1/sqrt vs the kernel's
    # hardware approx legitimately lands here).
    sparse = g.bf16_ulp_step(base, +1, fraction=1 / 1024, seed=0)
    r = g.gate_bf16_numerics(sparse, base, "t")
    assert r["passed"], r
    assert r["max_ulp"] == 1
    assert 0 < r["n_mismatch"] <= base.numel() // 1024 + 1
    assert r["cosine"] >= g.COS_LIMIT
    # uniform 1-ulp shift: a systematic 2^-8-relative bias is a bug even
    # though every element is within the ulp limit — the cosine gate
    # catches it.
    uniform = g.bf16_ulp_step(base, +1, fraction=1.0, seed=0)
    r = g.gate_bf16_numerics(uniform, base, "t")
    assert not r["passed"]
    assert r["max_ulp"] == 1 and r["cosine"] < g.COS_LIMIT
    # gross corruption and sign flips fail outright.
    gross = (base.to(torch.float32) * 1.01).to(torch.bfloat16)
    assert not g.gate_bf16_numerics(gross, base, "t")["passed"]
    assert not g.gate_bf16_numerics(-base, base, "t")["passed"]
    # one-sided NaN fails on finiteness, never silently.
    nan = base.clone()
    nan.view(-1)[0] = float("nan")
    r = g.gate_bf16_numerics(nan, base, "t")
    assert not r["passed"] and "finiteness" in r["reason"]
    # shape/dtype mismatches fail with a reason, not an exception.
    r = g.gate_bf16_numerics(base[:32], base, "t")
    assert not r["passed"] and "expected equal-shape" in r["reason"]
    r = g.gate_bf16_numerics(base.to(torch.float32), base, "t")
    assert not r["passed"]


def test_gate_bytes_exact_semantics() -> None:
    """Gate (b): byte-identical passes; a single flipped nibble or scale
    byte fails with the exact mismatch count and first offset."""
    g = _gates_module()
    gen = torch.Generator().manual_seed(0)
    data = torch.randint(0, 256, (64, 128), generator=gen,
                         dtype=torch.uint8)
    r = g.gate_bytes_exact(data, data, "t")
    assert r["passed"] and r["n_mismatch"] == 0
    assert r["first_mismatch_offset"] == -1
    one = data.clone()
    one.view(-1)[3] ^= 0x01  # a single flipped nibble
    r = g.gate_bytes_exact(one, data, "t")
    assert not r["passed"]
    assert r["n_mismatch"] == 1 and r["first_mismatch_offset"] == 3
    assert r["mismatch_fraction"] == 1 / data.numel()
    two = data.clone()
    two.view(-1)[7] ^= 0xF0
    two.view(-1)[8111] ^= 0x80
    r = g.gate_bytes_exact(two, data, "t")
    assert not r["passed"] and r["n_mismatch"] == 2


def test_gate_e4m3_codes_semantics() -> None:
    """Gate (c): sparse 1-code-step noise passes; a 3-code-step jump and a
    one-sided NaN code fail; the distance is measured in
    representable-value steps."""
    g = _gates_module()
    gen = torch.Generator().manual_seed(0)
    # valid finite magnitude codes only (0x00..0x7E, no NaN code 0x7F).
    base = torch.randint(0, 0x7F, (4096,), generator=gen,
                        dtype=torch.uint8)
    r = g.gate_e4m3_codes(base, base, "t")
    assert r["passed"] and r["max_code_distance"] == 0
    sparse = g.e4m3_code_step(base, +1, fraction=1 / 4096, seed=0)
    r = g.gate_e4m3_codes(sparse, base, "t")
    assert r["passed"], r
    assert r["max_code_distance"] == 1 and r["n_mismatch"] == 1
    jump = g.e4m3_code_step(base, +3, fraction=1 / 4096, seed=0)
    r = g.gate_e4m3_codes(jump, base, "t")
    assert not r["passed"] and "code distance" in r["reason"]
    nan = base.clone()
    nan.view(-1)[0] = 0x7F  # the NaN code on one side only
    r = g.gate_e4m3_codes(nan, base, "t")
    assert not r["passed"] and "NaN" in r["reason"]


def test_rz_rounded_product_is_true_rz() -> None:
    """ref.rz_rounded_product is exactly RZ of the exact product: it
    never moves the result away from zero past the exact product, it
    equals the RN product wherever RN rounded down, it differs from RN
    exactly on the upward-rounded elements, and pack_q4_row bytes are
    unchanged (the factor-out is behavior-preserving)."""
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(4096, 16, generator=gen) * 3
    c = torch.rand(4096, 16, generator=gen) * 4 + 0.5
    rz = ref.rz_rounded_product(x, c)
    rn = x * c
    exact = x.to(torch.float64) * c.to(torch.float64)
    # (a) RZ never exceeds the exact product in magnitude.
    signed = (rz.to(torch.float64) - exact) * torch.sign(exact)
    assert (signed <= 0).all()
    # (b) equals RN wherever RN did not land on the away-from-zero side;
    # differs from RN exactly on the away-side elements, for BOTH signs
    # (the correction is live, not vacuous, and symmetric).
    away = rn.to(torch.float64).abs() > exact.abs()
    assert torch.equal(rz == rn, ~away)
    n_fix = int((rz != rn).sum())
    n_fix_neg = int(((rz != rn) & (exact < 0)).sum())
    assert n_fix > 0, "the RZ correction never triggers on this grid"
    assert n_fix_neg > 0, "the RZ correction is asymmetric (sign bug)"
    # (c) differential: the PRE-FIX signed condition (p > exact, the
    # original inline emulation) violates the RZ magnitude property on
    # negative products that rounded away from zero; the shipped
    # implementation must not.
    safe_zero = torch.zeros_like(rn)
    pre_fix = torch.where(
        (rn.to(torch.float64) > exact) & torch.isfinite(rn),
        torch.nextafter(rn, safe_zero),
        rn,
    )
    signed_pre = (
        (pre_fix.to(torch.float64) - exact) * torch.sign(exact)
    )
    assert (signed_pre > 0).any(), (
        "the pre-fix bug is not reproduced on this grid"
    )
    assert not (signed > 0).any(), (
        "the shipped rz_rounded_product exceeds the exact magnitude"
    )


def test_kernel_pack_quotient_mirrors_the_reference_rz() -> None:
    """The Triton packer must compute the payload quotient through
    _rz_product (the bit-for-bit mirror of ref.rz_rounded_product), not a
    plain RN product."""
    root = Path(__file__).resolve().parents[1]
    kernels = (
        root / "src" / "vllm_exl3" / "minimax_q4kv_kernels.py"
    ).read_text()
    assert "def _rz_product(x, c):" in kernels
    assert "q = _rz_product(xb, 1.0 / s_vec)" in kernels
    assert "q = xb * (1.0 / s_vec)" not in kernels
    assert "rz_rounded_product" in kernels  # pinned to the reference by name


def test_harness_packing_gate_uses_the_kernels_own_rows() -> None:
    """Unit 1b must read the kernel's OWN processed rows back from qkv
    and run the reference packer on them (packing byte-exactness decoupled
    from norm rounding); unit 1a must gate q_out and the processed K and
    index_k rows; unit 1c must gate the fp8 index bytes."""
    root = Path(__file__).resolve().parents[1]
    harness = (root / "tools" / "gb10_q4kv_parity.py").read_text()

    def body(marker: str, stop: str) -> str:
        return harness[harness.index(marker):harness.index(stop)]

    unit_1b = body("def unit_1b", "def unit_1c")
    assert "slots_after()" in unit_1b  # the kernel's processed K readback
    assert "_scatter_q4_tokens" in unit_1b  # the reference packer
    assert "gate_bytes_exact" in unit_1b
    unit_1a = body("def unit_1a", "def unit_1b")
    for label in ("q_out", "processed-K", "processed-index_k"):
        assert f'"{label}"' in unit_1a
    assert "gate_bf16_numerics" in unit_1a
    unit_1c = body("def unit_1c", "def unit_1d")
    for label in ("index_q bytes", "index_k scatter bytes"):
        assert f'"{label}"' in unit_1c
    assert "gate_e4m3_codes" in unit_1c


def test_cpu_dry_run_exercises_the_new_gates() -> None:
    """--cpu-dry-run must self-test the gates on synthetic perturbations:
    a sparse 1-ulp perturbation and a 1-code-step perturbation must PASS,
    a single flipped nibble / scale byte must FAIL, and the reference
    paths must still run to completion."""
    root = Path(__file__).resolve().parents[1]
    harness = (root / "tools" / "gb10_q4kv_parity.py").read_text()
    for name in (
        "dry/0-w1-reference-runs",
        "dry/1a-gate-numerics-selftest",
        "dry/1b-gate-bytes-selftest",
        "dry/1c-gate-index-codes-selftest",
        "dry/2-decode-reference-runs",
        "dry/3-pool-bytes",
    ):
        assert f'("{name}"' in harness, f"missing dry-run unit {name}"
    assert "bf16_ulp_step" in harness  # the 1-ulp perturbation is exercised
    assert "e4m3_code_step" in harness
    assert "^= 0x01" in harness  # single flipped nibble/scale byte must fail
