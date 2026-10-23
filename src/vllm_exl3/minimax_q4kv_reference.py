"""Reference packer/unpacker and dispatch validation for the MiniMax-M3 q4 KV cache.

Milestone 1 (main MSA KV only; the indexer cache stays fp8 e4m3). This module
is pure torch: no vllm and no triton imports, so it runs on the CPU-only red
venv and is the byte-level specification the Triton kernels (W1/R2) implement.

Pool layout (upstream nvfp4 KV layout, one pool page = 128 tokens):

* pool page uint8 container ``[2 * num_kv_heads, 128, 72]``; slot
  ``2 * head + side`` (K = side 0, V = side 1) is one contiguous
  ``128 * 72 = 9216`` byte region (csrc requires ``stride(2) == 72`` and
  ``stride(1) == 9216`` so the region is addressable as flat bytes).
* inside each 9216-byte slot region: a data block ``[128 tokens, 64 B]``
  first, then a scale block ``[128 tokens, 8 B]``.
* data byte ``j`` of a token holds elements ``(2j, 2j+1)`` of the 128-dim
  vector as e2m1 nibbles, low nibble = even element.
* scale byte ``g`` of a token is the e4m3 group scale of elements
  ``[16g, 16g+16)``. K scales sit linearly at ``token * 8 + g``; V scales
  use the SM100 trtllm-gen 4x4 token-quad swizzle (see
  :func:`scale_block_offset`).

Quantization semantics (unit global scale, matching the vendor's shipped
nvfp4 KV path for this model):

1. the 128 values are rounded to the model dtype (bf16) first;
2. per 16-value group, ``s = max(amax / 6, 2^-9)`` encoded as saturating
   RNE e4m3 (NaN group entries are ignored by the max, as hardware fmax
   does);
3. payload ``q = rz(x * rn(1 / s_value))``;
4. code = saturating round-to-nearest-even e2m1 code of ``q`` (ties to the
   even code), negative zero stored as +0, NaN -> code 7 with its sign bit.

Decode is exact bit math (no fp8 dtype needed). The e4m3 encoder here is
cross-validated against torch's own ``float8_e4m3fn`` cast (which is the
saturating RNE conversion) in the CPU tests, and the e2m1 code function
against the upstream decision boundaries.
"""

from __future__ import annotations

import torch

HEAD_DIM = 128
GROUP_SIZE = 16
NUM_GROUPS = HEAD_DIM // GROUP_SIZE  # 8
DATA_BYTES = HEAD_DIM // 2  # 64
SCALE_BYTES = HEAD_DIM // 16  # 8
ROW_BYTES = DATA_BYTES + SCALE_BYTES  # 72 (container row width only)
POOL_BLOCK_SIZE = 128  # one sparse block == one pool page
SLOT_REGION = POOL_BLOCK_SIZE * ROW_BYTES  # 9216 flat bytes per slot
DATA_REGION = POOL_BLOCK_SIZE * DATA_BYTES  # 8192
SCALE_REGION = POOL_BLOCK_SIZE * SCALE_BYTES  # 1024


def nvfp4_row_bytes(head_dim: int) -> int:
    """Packed container width per (slot, token): e2m1 data + e4m3 scales."""
    return head_dim // 2 + head_dim // 16


def q4_pool_page_bytes(num_kv_heads: int, block_size: int = POOL_BLOCK_SIZE) -> int:
    """Bytes of one pool page: 2 slots/head (K and V) x block_size x 72."""
    if block_size != POOL_BLOCK_SIZE:
        raise ValueError(f"q4 KV pool requires block_size {POOL_BLOCK_SIZE}")
    return 2 * num_kv_heads * block_size * ROW_BYTES


def q4_pool_bytes_per_token(num_kv_heads: int) -> int:
    """Bytes of the main q4 KV pool per token per layer (576 B for 4 heads)."""
    return 2 * num_kv_heads * ROW_BYTES


def scale_block_offset(side: int, token: int, group: int) -> int:
    """Offset, within a slot's 1024-byte scale block, of the group-16 scale.

    K (side 0) scales are linear; V (side 1) scales use the SM100 trtllm-gen
    4x4 token-quad swizzle over (token, group).
    """
    if side == 0:
        return token * SCALE_BYTES + group
    if side != 1:
        raise ValueError(f"q4 KV side must be 0 (K) or 1 (V), got {side}")
    groups_per_quad = SCALE_BYTES // 4  # 2
    swizzled_token = (token // 4) * 4 + group // groups_per_quad
    swizzled_scale = (group % groups_per_quad) * 4 + token % 4
    return swizzled_token * SCALE_BYTES + swizzled_scale


def _scale_offset_table(side: int, tokens: int) -> torch.Tensor:
    """[tokens, NUM_GROUPS] offsets into the 1024-byte scale block."""
    t = torch.arange(tokens, dtype=torch.int64)
    g = torch.arange(NUM_GROUPS, dtype=torch.int64)
    if side == 0:
        return t[:, None] * SCALE_BYTES + g[None, :]
    swizzled_token = (t[:, None] // 4) * 4 + g[None, :] // 2
    swizzled_scale = (g[None, :] % 2) * 4 + (t[:, None] % 4)
    return swizzled_token * SCALE_BYTES + swizzled_scale


def flat_page_to_container(flat: torch.Tensor) -> torch.Tensor:
    """Map flat slot-region byte offsets to (row, col) of the [128, 72] view.

    The pool tensor is view-shaped ``[slots, 128, 72]`` but the semantic
    layout is flat bytes per slot region; this maps a flat offset to the
    container coordinates (row = flat // 72, col = flat % 72).
    """
    flat = flat.to(torch.int64)
    return torch.stack((flat // ROW_BYTES, flat % ROW_BYTES), dim=-1)


# ---------------------------------------------------------------------------
# Exact bit-level decode (no fp8 dtype required)
# ---------------------------------------------------------------------------


def e2m1_decode(nibbles: torch.Tensor) -> torch.Tensor:
    """e2m1 nibbles (uint8, 0..15) -> float32 values.

    Codes 0..7 decode to {0, 0.5, 1, 1.5, 2, 3, 4, 6}: e2m1 has no NaN
    (cvt.rn.satfinite.e2m1x2 saturates at +-6, vendor csrc e2m1Code). Bit 3
    is the sign.
    """
    n = nibbles.to(torch.int32) & 0xF
    code = n & 7
    low = code.to(torch.float32) * 0.5  # codes {0, 1}
    exp = (code >> 1).to(torch.float32) - 1.0
    high = torch.exp2(exp) * (1.0 + (code & 1).to(torch.float32) * 0.5)
    mag = torch.where(code < 2, low, high)
    mag = torch.where(code == 0, torch.zeros_like(mag), mag)
    return torch.where((n & 8) != 0, -mag, mag)


def e4m3_decode(bytes_: torch.Tensor) -> torch.Tensor:
    """e4m3fn bytes (uint8) -> float32 (0x7F/0xFF are NaN)."""
    b = bytes_.to(torch.int32) & 0xFF
    exp = (b >> 3) & 0xF
    man = b & 7
    sub = man.to(torch.float32) * (2.0**-9)
    normal = (man.to(torch.float32) + 8.0) * torch.exp2(
        exp.to(torch.float32) - 10.0
    )
    val = torch.where(exp == 0, sub, normal)
    val = torch.where((b & 0x7F) == 0x7F, torch.full_like(val, float("nan")), val)
    return torch.where((b & 0x80) != 0, -val, val)


# ---------------------------------------------------------------------------
# Exact bit-level encode (cross-validated against torch's satfinite cast)
# ---------------------------------------------------------------------------

_E4M3_MAX_CODE = 0x7E  # 448, the largest finite e4m3 code
_E4M3_NAN_CODE = 0x7F


def e4m3_encode_satfinite(x: torch.Tensor) -> torch.Tensor:
    """float32 -> e4m3fn bytes with saturating RNE rounding.

    Mirrors the hardware ``cvt.rn.satfinite`` conversion that torch's own
    ``.to(torch.float8_e4m3fn)`` implements: overflow saturates to +-448,
    NaN maps to the NaN code with its sign bit, +-0 keeps its sign.
    """
    shape = x.shape
    v = x.reshape(-1).to(torch.float32)
    sign = torch.signbit(v)
    ax = torch.where(torch.isnan(v), torch.zeros_like(v), v.abs())
    bits = ax.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    exp_f = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    # Target e4m3 exponent field (fp32 bias 127 -> e4m3 bias 7).
    e8 = exp_f - 120
    # Normal candidate: keep 3 mantissa bits, RNE on the dropped 20 bits.
    m3 = man >> 20
    rest = man & 0xFFFFF
    round_up = (rest > 0x80000) | ((rest == 0x80000) & ((m3 & 1) == 1))
    m3 = m3 + round_up.to(torch.int64)
    carry = m3 == 8
    m3 = torch.where(carry, torch.zeros_like(m3), m3)
    e8 = e8 + carry.to(torch.int64)
    normal_code = (e8 << 3) | m3
    normal_ok = (e8 >= 1) & (e8 <= 15) & ~((e8 == 15) & (m3 == 7))
    # Subnormal candidate: value = m * 2^-9, m = RNE integer of ax * 512.
    t = ax * 512.0
    fl = torch.floor(t)
    frac = t - fl
    fl_int = fl.to(torch.int64)
    inc = (frac > 0.5) | ((frac == 0.5) & ((fl_int & 1) == 1))
    m = fl + inc.to(torch.float32)
    # m == 8 carries into the smallest normal (2^-6, code 0x08).
    sub_code = torch.where(m == 8, torch.full_like(m, 8.0), m).to(torch.int64)
    sub_ok = e8 <= 0
    code = torch.where(sub_ok, sub_code, normal_code)
    ok = torch.where(sub_ok, sub_ok, normal_ok)
    code = torch.where(ok, code, torch.full_like(code, _E4M3_MAX_CODE))
    code = torch.where(torch.isnan(v), torch.full_like(code, _E4M3_NAN_CODE), code)
    byte = (code | (sign.to(torch.int64) << 7)) & 0xFF
    return byte.to(torch.uint8).reshape(shape)


_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e2m1_code(mag: torch.Tensor) -> torch.Tensor:
    """Saturating RNE e2m1 code (0..7) of |q|; NaN -> code 7.

    Ties round to the even code: the decision thresholds are the e2m1
    midpoints {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}; the tie at each
    lower midpoint stays below (0.25->0, 1.25->1, 2.5->2, 5->4) and the tie
    at each upper midpoint moves up (0.75->1, 1.75->2, 3.5->4).
    """
    a = mag.to(torch.float32)
    code = (
        (a > 0.25).to(torch.uint8)
        + (a >= 0.75).to(torch.uint8)
        + (a > 1.25).to(torch.uint8)
        + (a >= 1.75).to(torch.uint8)
        + (a > 2.5).to(torch.uint8)
        + (a >= 3.5).to(torch.uint8)
        + (a > 5.0).to(torch.uint8)
    )
    return torch.where(torch.isnan(a), torch.full_like(code, 7), code)


def e2m1_code_values() -> tuple[float, ...]:
    return _E2M1_VALUES


# ---------------------------------------------------------------------------
# Row pack / unpack (one 128-dim vector <-> 64 data bytes + 8 scale bytes)
# ---------------------------------------------------------------------------


def pack_q4_row(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack [..., 128] values into ([..., 64] data bytes, [..., 8] scale bytes).

    Implements the unit-global-scale nvfp4 semantics described in the module
    docstring. Inputs are rounded to bf16 first (the values the unfused path
    would read back), as the vendor kernel does.
    """
    if x.shape[-1] != HEAD_DIM:
        raise ValueError(f"q4 rows must be {HEAD_DIM}-dim, got {x.shape[-1]}")
    xb = x.to(torch.bfloat16).to(torch.float32)
    grouped = xb.unflatten(-1, (NUM_GROUPS, GROUP_SIZE))
    # amax with hardware fmax NaN semantics: NaN entries are ignored.
    absx = torch.where(torch.isnan(grouped), torch.zeros_like(grouped), grouped.abs())
    amax = absx.max(dim=-1).values
    s = torch.clamp(amax / 6.0, min=2.0**-9)
    scale_bytes = e4m3_encode_satfinite(s)
    s_value = e4m3_decode(scale_bytes)
    reciprocal = 1.0 / s_value  # RN(1/s); s_value >= 2^-9 so this is finite
    # rz(x * rn(1/s)): emulate round-toward-zero of the fp32 product.
    product_rn = grouped * reciprocal.unsqueeze(-1)
    product_exact = (
        grouped.to(torch.float64) * reciprocal.unsqueeze(-1).to(torch.float64)
    )
    rounded_up = product_rn.to(torch.float64) > product_exact
    safe_zero = torch.zeros_like(product_rn)
    quotient = torch.where(
        rounded_up & torch.isfinite(product_rn),
        torch.nextafter(product_rn, safe_zero),
        product_rn,
    )
    codes = e2m1_code(quotient.abs())
    # Zero is stored as +0: clear the sign of every zero-magnitude nibble.
    signed = torch.where(
        torch.signbit(quotient), (codes | 8).to(torch.uint8), codes
    )
    nibbles = torch.where(codes == 0, torch.zeros_like(signed), signed)
    nibbles = nibbles.reshape(*x.shape[:-1], HEAD_DIM)
    low, high = nibbles[..., 0::2], nibbles[..., 1::2]
    data_bytes = (low | (high << 4)).to(torch.uint8)
    return data_bytes, scale_bytes


def unpack_q4_row(
    data_bytes: torch.Tensor, scale_bytes: torch.Tensor
) -> torch.Tensor:
    """([..., 64] data, [..., 8] scales) -> [..., 128] float32 values.

    Data byte ``j`` holds elements ``(2j, 2j+1)`` as (low, high) nibbles.
    """
    if data_bytes.shape[-1] != DATA_BYTES or scale_bytes.shape[-1] != SCALE_BYTES:
        raise ValueError("q4 row unpack expects 64 data and 8 scale bytes")
    low = e2m1_decode(data_bytes & 0xF)
    high = e2m1_decode(data_bytes >> 4)
    values = torch.stack((low, high), dim=-1).reshape(
        *data_bytes.shape[:-1], HEAD_DIM
    )
    scales = e4m3_decode(scale_bytes)
    return values * scales.repeat_interleave(GROUP_SIZE, dim=-1)


# ---------------------------------------------------------------------------
# Slot-region pack / unpack (one slot = 128 tokens x 9216 flat bytes)
# ---------------------------------------------------------------------------


def pack_slot_region_side(
    rows: torch.Tensor, side: int
) -> torch.Tensor:
    """:func:`pack_slot_region` with the side (K = 0, V = 1) given explicitly."""
    tokens, dim = rows.shape
    if dim != HEAD_DIM:
        raise ValueError("slot rows are [tokens, 128]")
    if side not in (0, 1):
        raise ValueError(f"q4 KV side must be 0 or 1, got {side}")
    data, scales = pack_q4_row(rows)
    flat = torch.zeros((tokens * ROW_BYTES,), dtype=torch.uint8)
    flat[: tokens * DATA_BYTES].view(tokens, DATA_BYTES).copy_(data)
    offsets = _scale_offset_table(side, tokens).reshape(-1)
    flat[DATA_REGION:].scatter_(0, offsets, scales.reshape(-1))
    return flat


def unpack_slot_region_side(flat: torch.Tensor, side: int) -> torch.Tensor:
    """Flat uint8 [tokens*72] of one slot -> values [tokens, 128]."""
    tokens = flat.numel() // ROW_BYTES
    if side not in (0, 1):
        raise ValueError(f"q4 KV side must be 0 or 1, got {side}")
    data = flat[: DATA_REGION].view(tokens, DATA_BYTES)
    low = e2m1_decode(data & 0xF)
    high = e2m1_decode(data >> 4)
    values = torch.stack((low, high), dim=-1).reshape(tokens, HEAD_DIM)
    offsets = _scale_offset_table(side, tokens).reshape(-1)
    stored = flat[DATA_REGION:].gather(0, offsets).view(tokens, NUM_GROUPS)
    scales = e4m3_decode(stored)
    grouped = values.view(tokens, NUM_GROUPS, GROUP_SIZE)
    return (grouped * scales.unsqueeze(-1)).reshape(tokens, HEAD_DIM)


def pack_pool_page(
    rows: torch.Tensor,  # [2H, 128, 128] slot-major values
) -> torch.Tensor:
    """Pack a full pool page into flat uint8 [2H * 9216] (slot-major)."""
    slots, tokens, dim = rows.shape
    if dim != HEAD_DIM or tokens != POOL_BLOCK_SIZE:
        raise ValueError("pool pages are [slots, 128, 128]")
    flat = torch.zeros((slots * SLOT_REGION,), dtype=torch.uint8)
    for slot in range(slots):
        side = slot % 2
        flat[slot * SLOT_REGION : (slot + 1) * SLOT_REGION] = (
            pack_slot_region_side(rows[slot], side)
        )
    return flat


def unpack_pool_page(flat: torch.Tensor) -> torch.Tensor:
    """Flat uint8 [2H * 9216] slot-major -> values [2H, 128, 128]."""
    slots = flat.numel() // SLOT_REGION
    values = torch.zeros((slots, POOL_BLOCK_SIZE, HEAD_DIM), dtype=torch.float32)
    for slot in range(slots):
        values[slot] = unpack_slot_region_side(
            flat[slot * SLOT_REGION : (slot + 1) * SLOT_REGION], slot % 2
        )
    return values


def split_page_kv(page_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Slot-major page values [2H, 128, 128] -> (K [H, ...], V [H, ...])."""
    return page_values[0::2], page_values[1::2]


# ---------------------------------------------------------------------------
# Reference W1: Gemma QK-norm + partial NeoX RoPE + q4 pack (pure torch)
# ---------------------------------------------------------------------------


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma RMSNorm in fp32: x * rsqrt(mean(x^2) + eps) * (1 + w).

    Uses a precisely-rounded reciprocal square root (1/sqrt), not the
    approximate hardware rsqrt of the vendor kernel; the CPU tests pin this
    choice and the Triton kernel implements the same precisely-rounded form
    so W1 and this reference agree bit-exactly.
    """
    sumsq = (x.to(torch.float32) ** 2).sum(dim=-1, keepdim=True)
    r = 1.0 / torch.sqrt(sumsq / HEAD_DIM + eps)
    return x.to(torch.float32) * r * (1.0 + weight.to(torch.float32))


def rope_partial_neox(
    x: torch.Tensor, cos_sin_row: torch.Tensor
) -> torch.Tensor:
    """Partial NeoX RoPE on the leading rotary_dim dims of a 128-dim row.

    ``cos_sin_row`` is one row of the cos/sin cache:
    ``[cos(rot_dim/2) | sin(rot_dim/2)]``. Pairs (i, i + half) rotate as
    y[i] = x[i]*cos[i] - x[i+half]*sin[i],
    y[i+half] = x[i+half]*cos[i] + x[i]*sin[i]; the trailing dims pass
    through unchanged.
    """
    rot_dim = int(cos_sin_row.shape[-1])
    half = rot_dim // 2
    if rot_dim % 2 or rot_dim > HEAD_DIM:
        raise ValueError("rotary_dim must be even and <= 128")
    d = torch.arange(HEAD_DIM, dtype=torch.int64)
    in_rope = d < rot_dim
    cos_idx = torch.where(d < half, d, d - half).clamp(max=half - 1)
    sin_idx = (cos_idx + half).clamp(max=rot_dim - 1)
    partner = torch.where(d < half, d + half, d - half)
    cos = cos_sin_row.to(torch.float32)[..., cos_idx]
    sin = cos_sin_row.to(torch.float32)[..., sin_idx]
    sign = torch.where(d < half, -1.0, 1.0)
    partner_x = x.gather(-1, partner.expand(*x.shape[:-1], HEAD_DIM))
    if x.dim() > 2:
        # Broadcast one cos/sin row over the leading head dimension.
        cos = cos.view(*cos.shape[:-1], *([1] * (x.dim() - 2)), HEAD_DIM)
        sin = sin.view(*sin.shape[:-1], *([1] * (x.dim() - 2)), HEAD_DIM)
    rotated = x * cos + sign * partner_x * sin
    return torch.where(in_rope, rotated, x)


def reference_w1_write(
    qkv_rows: torch.Tensor,  # [N, (nq + 2*nkv + niq + 1) * 128] bf16/fp32
    q_norm_w: torch.Tensor,  # [128]
    k_norm_w: torch.Tensor,  # [128]
    iq_norm_w: torch.Tensor,  # [128]
    ik_norm_w: torch.Tensor,  # [128]
    cos_sin_cache: torch.Tensor,  # [max_pos, rotary_dim]
    positions: torch.Tensor,  # [N] int64
    eps: float,
    rotary_dim: int,
    num_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    slot_mapping: torch.Tensor,  # [N] int64, -1 skips
    index_slot_mapping: torch.Tensor,  # [N] int64, -1 skips
) -> dict:
    """Full reference of the q4 write path (W1) for one batch of tokens.

    Returns a dict with the bf16 ``q`` buffer, the e4m3 ``index_q`` bytes
    per token, the packed main pool ``[num_pages, 2*nkv, 9216]`` uint8
    (slot-major flat bytes; slot ``2 * head + side``) and the e4m3
    ``index_k`` bytes per token (the caller scatters them into the fp8
    index cache at ``index_slot_mapping * 128``).
    """
    num_tokens = qkv_rows.shape[0]
    row = qkv_rows.shape[1]
    niq = num_index_heads
    expected = (num_heads + 2 * num_kv_heads + niq + 1) * HEAD_DIM
    if row != expected:
        raise ValueError(f"qkv rows must be {expected} wide, got {row}")
    if positions.dim() != 1 or positions.shape[0] != num_tokens:
        raise ValueError(
            f"positions must be [{num_tokens}], got {tuple(positions.shape)}"
        )
    if cos_sin_cache.dim() != 2 or cos_sin_cache.shape[-1] < rotary_dim:
        raise ValueError(
            "cos_sin_cache must be [rows, rotary_dim] with rows covering "
            f"every position; got {tuple(cos_sin_cache.shape)} for "
            f"rotary_dim {rotary_dim}"
        )
    pos_lo = int(positions.min())
    pos_hi = int(positions.max())
    if pos_lo < 0 or pos_hi >= cos_sin_cache.shape[0]:
        raise ValueError(
            f"positions [{pos_lo}, {pos_hi}] exceed the cos_sin_cache "
            f"({cos_sin_cache.shape[0]} rows): the rotary cache must cover "
            f"max(positions) + 1 rows. This is an input-data bug in the "
            f"caller (build the cache large enough), not a kernel bug."
        )
    x = qkv_rows.to(torch.float32).view(num_tokens, -1, HEAD_DIM)
    q_out = torch.empty((num_tokens, num_heads, HEAD_DIM), dtype=torch.bfloat16)
    index_q_bytes = (
        torch.empty((num_tokens, niq, HEAD_DIM), dtype=torch.uint8)
        if niq
        else torch.zeros((num_tokens, 0, HEAD_DIM), dtype=torch.uint8)
    )
    num_pages = 0
    if slot_mapping.numel():
        valid_sm = slot_mapping[slot_mapping >= 0]
        if valid_sm.numel():
            num_pages = int((valid_sm.max().item()) // POOL_BLOCK_SIZE) + 1
    pool = torch.zeros(
        (num_pages, 2 * num_kv_heads, SLOT_REGION), dtype=torch.uint8
    )
    index_k_bytes = torch.empty((num_tokens, HEAD_DIM), dtype=torch.uint8)
    # Branch slices of the fused row: [q | k | v | index_q | index_k].
    q_slice = x[:, :num_heads]
    k_slice = x[:, num_heads : num_heads + num_kv_heads]
    v_slice = x[:, num_heads + num_kv_heads : num_heads + 2 * num_kv_heads]
    iq_slice = x[:, num_heads + 2 * num_kv_heads : num_heads + 2 * num_kv_heads + niq]
    ik_slice = x[:, num_heads + 2 * num_kv_heads + niq :]
    cos_sin = cos_sin_cache.to(torch.float32)[positions]

    # Q: Gemma norm + RoPE -> bf16 q buffer.
    qn = gemma_rmsnorm(q_slice, q_norm_w, eps)
    qn = rope_partial_neox(qn, cos_sin[:, :rotary_dim])
    q_out.copy_((qn.to(torch.bfloat16)))

    # K: Gemma norm + RoPE -> q4 pack into slot 2*head + 0.
    kn = rope_partial_neox(
        gemma_rmsnorm(k_slice, k_norm_w, eps), cos_sin[:, :rotary_dim]
    )
    # V: no norm, no RoPE -> q4 pack into slot 2*head + 1.
    for head in range(num_kv_heads):
        pool[:, 2 * head] = _scatter_q4_tokens(
            kn[:, head], slot_mapping, num_pages, 2 * head
        )
        pool[:, 2 * head + 1] = _scatter_q4_tokens(
            v_slice[:, head], slot_mapping, num_pages, 2 * head + 1
        )

    # index_q: Gemma norm + RoPE -> e4m3(bf16(v)) satfinite bytes.
    if niq:
        iq = rope_partial_neox(
            gemma_rmsnorm(iq_slice, iq_norm_w, eps), cos_sin[:, :rotary_dim]
        )
        index_q_bytes.copy_(
            e4m3_encode_satfinite(iq.to(torch.bfloat16).to(torch.float32))
        )

    # index_k: Gemma norm + RoPE -> e4m3(clamp(v, +-448)) bytes (fp8 cache).
    ik = rope_partial_neox(
        gemma_rmsnorm(ik_slice[:, 0], ik_norm_w, eps), cos_sin[:, :rotary_dim]
    )
    index_k_bytes.copy_(_e4m3_clamped(ik))
    return {
        "q": q_out,
        "index_q": index_q_bytes,
        "pool": pool,  # [num_pages, 2*nkv, 9216] uint8 (slot-major, flat)
        "index_k": index_k_bytes,  # [N, 128] e4m3 bytes (fp8 index cache)
    }


def _scatter_q4_tokens(
    values: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_pages: int,
    slot: int,
) -> torch.Tensor:
    """Scatter packed per-token rows into a slot's flat page regions.

    Returns uint8 [num_pages, 9216]: token ``sm`` writes its 64 data bytes
    at flat ``(sm % 128) * 64 + j`` and its 8 scale bytes at
    ``8192 + scale_block_offset(side, sm % 128, g)`` of page ``sm // 128``.
    Tokens with slot ``-1`` are skipped.
    """
    side = slot % 2
    data, scales = pack_q4_row(values)  # [N, 64], [N, 8]
    out = torch.zeros((num_pages, SLOT_REGION), dtype=torch.uint8)
    valid = slot_mapping >= 0
    if not valid.any():
        return out
    sm = slot_mapping[valid]
    page = (sm // POOL_BLOCK_SIZE).to(torch.int64)
    tok = (sm % POOL_BLOCK_SIZE).to(torch.int64)
    base = (page * SLOT_REGION).unsqueeze(1)
    data_pos = tok.unsqueeze(1) * DATA_BYTES + torch.arange(
        DATA_BYTES, dtype=torch.int64
    )
    out.view(-1)[(base + data_pos).reshape(-1)] = data[valid].reshape(-1)
    offsets = _scale_offset_table(side, POOL_BLOCK_SIZE)  # [128, 8]
    scale_pos = DATA_REGION + offsets[tok]  # [V, 8]
    out.view(-1)[(base + scale_pos).reshape(-1)] = scales[valid].reshape(-1)
    return out


def _e4m3_clamped(x: torch.Tensor) -> torch.Tensor:
    """e4m3 bytes of clamp(x, +-448) with fmax/fmin NaN semantics."""
    v = x.to(torch.float32)
    v = torch.where(torch.isnan(v), torch.full_like(v, -448.0), v)
    return e4m3_encode_satfinite(torch.clamp(v, -448.0, 448.0))


# ---------------------------------------------------------------------------
# Dispatch validation (fail closed; shared by the overlay wrapper and tests)
# ---------------------------------------------------------------------------


def validate_w1_call(
    *,
    qkv_dtype: str,
    qkv_width: int,
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    rotary_dim: int,
    block_size: int,
    pool_dtype: str,
    pool_dim: int,
    pool_slots: int,
    pool_tokens: int,
    pool_width: int,
    pool_row_stride: int,
    pool_slot_stride: int,
    q_out_present: bool,
    q_fp8_out_present: bool,
    index_cache_dtype: str | None,
    index_q_out_dtype: str | None,
    kv_k_scale: float | None,
    kv_v_scale: float | None,
) -> list[str]:
    """Return the list of reasons the q4 W1 path must refuse this call.

    An empty list means the call is supported by milestone 1: bf16 rows,
    unit global scales, fp8 e4m3 indexer cache, block size 128.
    """
    errors: list[str] = []
    if qkv_dtype != "bfloat16":
        errors.append(f"q4 W1 requires bfloat16 qkv rows, got {qkv_dtype}")
    expected_width = (
        (num_heads + 2 * num_kv_heads + num_index_heads + 1) * HEAD_DIM
        if num_index_heads > 0
        else (num_heads + 2 * num_kv_heads) * HEAD_DIM
    )
    if qkv_width != expected_width:
        errors.append(
            f"q4 W1 qkv width {qkv_width} does not match the branch layout "
            f"{expected_width}"
        )
    if num_tokens <= 0:
        errors.append("q4 W1 requires a positive token count")
    if not (0 < rotary_dim <= HEAD_DIM and rotary_dim % 8 == 0):
        errors.append(
            f"q4 W1 requires 0 < rotary_dim <= 128 and rotary_dim % 8 == 0, "
            f"got {rotary_dim}"
        )
    if block_size != POOL_BLOCK_SIZE:
        errors.append(
            f"q4 W1 requires block_size {POOL_BLOCK_SIZE}, got {block_size}"
        )
    if pool_dtype != "uint8":
        errors.append(f"q4 pool must be uint8, got {pool_dtype}")
    if pool_dim != 4 or pool_slots != 2 * num_kv_heads:
        errors.append(
            "q4 pool must be [pages, 2*num_kv_heads, block_size, 72], got "
            f"dim {pool_dim} with {pool_slots} slots"
        )
    if pool_tokens != POOL_BLOCK_SIZE or pool_width != ROW_BYTES:
        errors.append(
            f"q4 pool page must be [*, 128, 72], got [*, {pool_tokens}, "
            f"{pool_width}]"
        )
    if pool_row_stride != ROW_BYTES or pool_slot_stride != SLOT_REGION:
        errors.append(
            "q4 pool slot regions must be contiguous "
            f"(strides {pool_row_stride}/{pool_slot_stride}, expected "
            f"{ROW_BYTES}/{SLOT_REGION})"
        )
    if q_fp8_out_present:
        errors.append(
            "q4 W1 does not emit an e4m3 q (the CUTLASS decode path); "
            "q_fp8_out must be None"
        )
    if not q_out_present:
        errors.append("q4 W1 requires the bf16 q_out buffer")
    # Milestone 1: the indexer cache stays fp8 e4m3 (uint8 means nvfp4 pages).
    if num_index_heads > 0 and index_cache_dtype is not None:
        if index_cache_dtype == "uint8":
            errors.append(
                "q4 (nvfp4) indexer cache is milestone 2; the indexer cache "
                "must stay fp8 e4m3"
            )
        elif index_cache_dtype != "float8_e4m3fn":
            errors.append(
                f"q4 W1 requires the fp8 e4m3 indexer cache, got "
                f"{index_cache_dtype}"
            )
    if num_index_heads > 0 and index_q_out_dtype is not None:
        if index_q_out_dtype != "float8_e4m3fn":
            errors.append(
                f"q4 W1 requires the e4m3 index_q buffer, got "
                f"{index_q_out_dtype}"
            )
    for name, scale in (("kv_k_scale", kv_k_scale), ("kv_v_scale", kv_v_scale)):
        if scale is not None and scale != 1.0:
            errors.append(
                f"q4 W1 milestone 1 implements the unit global scale only; "
                f"{name} must be None or 1.0, got {scale}"
            )
    return errors


# ---------------------------------------------------------------------------
# Prefill staging: dequantize the selected q4 pages for the eager prefill path
# ---------------------------------------------------------------------------


def dequant_pool_pages(pool_flat: torch.Tensor) -> torch.Tensor:
    """Flat pool [pages, 2H, 9216] -> values [pages, 2H, 128, 128] fp32.

    Vectorized over every (page, slot) at once; the per-side scale offsets
    differ only by slot parity (even = K/linear, odd = V/swizzled).
    """
    pages, slots, region = pool_flat.shape
    if region != SLOT_REGION:
        raise ValueError("pool flat regions are 9216 bytes")
    data = pool_flat[:, :, :DATA_REGION].reshape(
        pages * slots, POOL_BLOCK_SIZE, DATA_BYTES
    )
    low = e2m1_decode(data & 0xF)
    high = e2m1_decode(data >> 4)
    values = torch.stack((low, high), dim=-1).reshape(
        pages, slots, POOL_BLOCK_SIZE, HEAD_DIM
    )
    scale_region = pool_flat[:, :, DATA_REGION:]  # [pages, slots, 1024]
    off = torch.stack(
        (
            _scale_offset_table(0, POOL_BLOCK_SIZE).reshape(-1),
            _scale_offset_table(1, POOL_BLOCK_SIZE).reshape(-1),
        )
    )  # [2, 1024]
    idx = off[torch.arange(slots, dtype=torch.int64) % 2]  # [slots, 1024]
    stored = scale_region.gather(
        2, idx.view(1, slots, -1).expand(pages, -1, -1)
    ).reshape(pages, slots, POOL_BLOCK_SIZE, NUM_GROUPS)
    scales = e4m3_decode(stored).repeat_interleave(GROUP_SIZE, dim=-1)
    return values * scales


def build_prefill_staging(
    pool: torch.Tensor,  # bound uint8 [nb, 2H, 128, 72]
    block_table: torch.Tensor,  # [num_reqs, max_blocks] int32/int64
    topk: torch.Tensor,  # [nkv, total_q, topk] block ids (logical)
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the bf16 staging pool for the eager prefill path.

    The selected logical blocks (any non-sentinel top-k entry in the batch)
    are resolved through the block table to physical pages; every distinct
    selected physical page is dequantized once into ``staging`` (bf16
    ``[sel + 1, H, 128, 256]``, K then V, plus one trailing zero page), and
    ``fake_block_table`` remaps every original block-table entry to its
    staging page (unselected -> the zero page). The eager prefill then runs
    the existing bf16 prefill kernel on the staging pool with the ORIGINAL
    top-k indices (their logical block ids drive causal masking, which is
    unchanged). This is the slow-but-correct milestone-1 prefill path.
    """
    nb, slots, tokens, width = pool.shape
    if (
        slots != 2 * num_kv_heads
        or tokens != POOL_BLOCK_SIZE
        or width != ROW_BYTES
    ):
        raise ValueError("pool must be [nb, 2*num_kv_heads, 128, 72] uint8")
    flat = pool.reshape(nb, slots, SLOT_REGION)
    blks = topk.reshape(-1).to(torch.int64)
    blks = blks[(blks >= 0) & (blks < block_table.shape[1])]
    selected = torch.empty(0, dtype=torch.int64)
    if blks.numel() > 0:
        cand = block_table[:, blks.unique()].reshape(-1).to(torch.int64)
        selected = cand[(cand >= 0) & (cand < nb)].unique()
    zero_page = int(selected.numel())
    staging = torch.zeros(
        (zero_page + 1, num_kv_heads, POOL_BLOCK_SIZE, 2 * HEAD_DIM),
        dtype=torch.bfloat16,
        device=pool.device,
    )
    if zero_page > 0:
        values = dequant_pool_pages(flat[selected])  # [P, 2H, 128, 128]
        k, v = values[:, 0::2], values[:, 1::2]  # [P, H, 128, 128]
        staging[:zero_page, :, :, :HEAD_DIM] = k.to(torch.bfloat16)
        staging[:zero_page, :, :, HEAD_DIM:] = v.to(torch.bfloat16)
    lookup = torch.full((nb,), zero_page, dtype=torch.int64, device=pool.device)
    if zero_page > 0:
        lookup[selected] = torch.arange(zero_page, device=pool.device)
    fake_block_table = lookup[
        block_table.clamp(min=0).reshape(-1).to(torch.int64)
    ].view_as(block_table)
    return staging, fake_block_table.to(block_table.dtype)


# ---------------------------------------------------------------------------
# Reference sparse attention (fp32 oracle for R2/prefill tests)
# ---------------------------------------------------------------------------


def reference_sparse_attention(
    q: torch.Tensor,  # [total_q, H, 128] fp32
    k_pages: torch.Tensor,  # [pages, H, 128, 128] fp32 (token, dim)
    v_pages: torch.Tensor,  # [pages, H, 128, 128] fp32
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    topk: torch.Tensor,  # [total_q, topk] logical block ids per query token
    query_pos: torch.Tensor,  # [total_q] absolute position of each query token
    sm_scale: float,
) -> torch.Tensor:
    """Dense fp32 block-sparse attention over the selected blocks.

    Every query token attends to the union of its top-k blocks restricted to
    its causal past (positions < query_pos + 1); softmax over the union.
    This is the correctness oracle, not a performance path.
    """
    total_q, num_heads, head_dim = q.shape
    num_kv_heads = k_pages.shape[1]
    group = num_heads // num_kv_heads
    out = torch.zeros((total_q, num_heads, head_dim), dtype=torch.float32)
    for i in range(total_q):
        pos = int(query_pos[i])
        blocks = sorted({int(b) for b in topk[i].tolist() if b >= 0})
        keys, vals, mask = [], [], []
        for b in blocks:
            page = int(block_table.reshape(-1)[b]) if block_table.numel() else 0
            if page < 0:
                continue
            for t in range(POOL_BLOCK_SIZE):
                abs_pos = b * POOL_BLOCK_SIZE + t
                if abs_pos > pos:
                    continue
                keys.append(k_pages[page, :, t])
                vals.append(v_pages[page, :, t])
                mask.append(abs_pos)
        if not keys:
            continue
        k = torch.stack(keys)  # [T, Hkv, 128]
        v = torch.stack(vals)
        scores = torch.zeros((num_heads, k.shape[0]), dtype=torch.float32)
        for h in range(num_heads):
            kvh = h // group
            scores[h] = (q[i, h] @ k[:, kvh].T) * sm_scale
        probs = torch.softmax(scores, dim=-1)
        for h in range(num_heads):
            kvh = h // group
            out[i, h] = probs[h] @ v[:, kvh]
    return out
