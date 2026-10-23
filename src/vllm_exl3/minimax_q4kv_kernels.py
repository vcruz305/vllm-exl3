"""Triton kernels for the MiniMax-M3 q4 KV cache (milestone 1, sm_121).

W1: fused Gemma QK-norm + partial NeoX RoPE + e2m1/e4m3 q4 pack +
scatter-insert into the main pool. The index_q/index_k outputs keep their
fp8 e4m3 semantics (the indexer cache stays fp8 in milestone 1), but the
fp8 buffers are handed to the kernels as uint8 byte views of the same
storage (see _fp8_bytes): every load/store in these kernels is raw byte
math, and a raw fp8 pointer would make Triton's frontend try to cast the
bytes to fp8e4nv and fail to compile (GB10 val1).

R2: q4-dequant split-K sparse decode producing partials compatible with
vLLM's existing merge kernel (reused unchanged).

The byte-level semantics are specified by
:mod:`vllm_exl3.minimax_q4kv_reference`; the encoders/decoders here
implement the same bit math in Triton and are parity-checked against that
reference on GB10. This module imports triton and is only imported on
CUDA paths.
"""

from __future__ import annotations

import torch

from vllm_exl3 import minimax_q4kv_reference as ref

_SLOT_REGION = 9216


def _import_triton():
    from vllm.triton_utils import tl, triton

    return tl, triton


def _build_kernels():
    tl, triton = _import_triton()

    # ------------------------------------------------------------------
    # shared bit math (mirrors minimax_q4kv_reference)
    # ------------------------------------------------------------------

    @triton.jit
    def _e4m3_mag_code(v):
        # Saturating RNE e4m3 magnitude code of v >= 0 (see the reference
        # encoder); the caller applies the sign bit.
        ax = tl.abs(v)
        bits = tl.cast(ax, tl.int32, bitcast=True).to(tl.int64)
        exp_f = (bits >> 23) & 0xFF
        man = bits & 0x7FFFFF
        e8 = exp_f - 120
        m3 = man >> 20
        rest = man & 0xFFFFF
        round_up = (rest > 0x80000) | ((rest == 0x80000) & ((m3 & 1) == 1))
        m3 = m3 + round_up.to(tl.int64)
        carry = m3 == 8
        m3 = tl.where(carry, 0, m3)
        e8 = e8 + carry.to(tl.int64)
        normal_code = (e8 << 3) | m3
        normal_ok = (e8 >= 1) & (e8 <= 15) & ~((e8 == 15) & (m3 == 7))
        t = ax * 512.0
        fl = tl.floor(t)
        frac = t - fl
        inc = (frac > 0.5) | ((frac == 0.5) & ((fl.to(tl.int64) & 1) == 1))
        m = fl + inc.to(tl.float32)
        sub_code = tl.where(m == 8.0, 8.0, m).to(tl.int64)
        use_sub = e8 <= 0
        code = tl.where(use_sub, sub_code, normal_code)
        ok = tl.where(use_sub, use_sub, normal_ok)
        code = tl.where(ok, code, 0x7E)
        code = tl.where(v != v, 0x7F, code)  # NaN -> the NaN code
        return code

    @triton.jit
    def _e2m1_code(mag):
        code = (
            (mag > 0.25).to(tl.int64)
            + (mag >= 0.75).to(tl.int64)
            + (mag > 1.25).to(tl.int64)
            + (mag >= 1.75).to(tl.int64)
            + (mag > 2.5).to(tl.int64)
            + (mag >= 3.5).to(tl.int64)
            + (mag > 5.0).to(tl.int64)
        )
        return tl.where(mag != mag, 7, code)

    @triton.jit
    def _e2m1_decode(nib):
        # e2m1 has no NaN: code 7 is the saturation value 6 (matches the
        # vendor csrc e2m1Code / cvt.rn.satfinite.e2m1x2).
        n = nib.to(tl.int64) & 0xF
        code = n & 7
        low = code.to(tl.float32) * 0.5
        exp = (code >> 1).to(tl.float32) - 1.0
        high = tl.exp2(exp) * (1.0 + (code & 1).to(tl.float32) * 0.5)
        mag = tl.where(code < 2, low, high)
        mag = tl.where(code == 0, 0.0, mag)
        return tl.where((n & 8) != 0, -mag, mag)

    @triton.jit
    def _e4m3_decode(byte):
        b = byte.to(tl.int64) & 0xFF
        exp = (b >> 3) & 0xF
        man = b & 7
        sub = man.to(tl.float32) * 0.001953125
        normal = (man.to(tl.float32) + 8.0) * tl.exp2(
            exp.to(tl.float32) - 10.0
        )
        val = tl.where(exp == 0, sub, normal)
        val = tl.where((b & 0x7F) == 0x7F, float("nan"), val)
        return tl.where((b & 0x80) != 0, -val, val)

    @triton.jit
    def _expand16(v):
        # [N, 8] -> [N, 128] with each group value repeated 16 times.
        a = tl.join(v, v)  # [N, 8, 2]
        a = tl.join(a, a)  # [N, 8, 2, 2]
        a = tl.join(a, a)  # [N, 8, 2, 2, 2]
        a = tl.join(a, a)  # [N, 8, 2, 2, 2, 2]
        return tl.reshape(a, (v.shape[0], 128))

    @triton.jit
    def _nibbles_to_bytes(nib):
        # [N, 128] nibbles -> [N, 64] bytes; byte j = code(2j) | code(2j+1)<<4.
        # Triton has no slicing, so pair via a weighted sum over the joined
        # minor axis: weights [1, 16] implement the nibble shift.
        pairs = tl.reshape(nib, (nib.shape[0], 64, 2))
        w = tl.arange(0, 2) * 15 + 1  # [1, 16]
        byte = tl.sum(pairs * w[None, None, :], axis=2)
        return byte

    # ------------------------------------------------------------------
    # W1
    # ------------------------------------------------------------------

    @triton.jit
    def _gemma_norm(x, w_ptr, eps):
        # Returns (normed, r); r is invariant under row permutation so the
        # rope partner values can be reconstructed from permuted loads.
        sumsq = tl.sum(x * x, axis=0)
        r = 1.0 / tl.sqrt(sumsq / 128.0 + eps)
        w = tl.load(w_ptr + tl.arange(0, 128)).to(tl.float32)
        return x * r * (1.0 + w), r

    @triton.jit
    def _rope(xn, r, row_ptr, w_ptr, cos_sin_ptr, pos, rotary_dim):
        # Partial NeoX rope over the first rotary_dim dims of the already
        # normed row xn; the partner values are normed via permutation-
        # invariant r and the norm weight at the partner index.
        d = tl.arange(0, 128)
        half = rotary_dim // 2
        in_rope = d < rotary_dim
        first = d < half
        cs_idx = tl.where(in_rope, tl.where(first, d, d - half), 0)
        c = tl.load(cos_sin_ptr + pos * rotary_dim + cs_idx).to(tl.float32)
        s = tl.load(cos_sin_ptr + pos * rotary_dim + half + cs_idx).to(
            tl.float32
        )
        partner = tl.where(in_rope, tl.where(first, d + half, d - half), d)
        pv_raw = tl.load(row_ptr + partner).to(tl.float32)
        pv_w = tl.load(w_ptr + partner).to(tl.float32)
        pv = pv_raw * r * (1.0 + pv_w)
        sign = tl.where(first, -1.0, 1.0)
        rotated = xn * c + sign * pv * s
        return tl.where(in_rope, rotated, xn)

    @triton.jit
    def _rz_product(x, c):
        # RZ(x * c) at fp32 precision, mirroring ref.rz_rounded_product
        # bit-for-bit: p is the fp32 RN product; when p landed on the
        # away-from-zero side of the exact (f64) product, step it one
        # representable value toward zero (sign-magnitude bit pattern
        # - 1) — for BOTH signs. Non-finite products pass through
        # unmodified (the e2m1 ladder saturates them), exactly as the
        # reference's isfinite guard does.
        p = x * c
        pd = x.to(tl.float64) * c.to(tl.float64)
        sign_pd = tl.where(pd < 0.0, -1.0, 1.0)
        away = (p.to(tl.float64) - pd) * sign_pd > 0.0
        finite = (p == p) & (p != float("inf")) & (p != float("-inf"))
        fix = away & finite
        bits = tl.cast(p, tl.int32, bitcast=True)
        bits = tl.where(fix, bits - 1, bits)
        return tl.cast(bits, tl.float32, bitcast=True)

    @triton.jit
    def _pack_q4(xb, tok, side):
        # xb: [128] fp32 (already bf16-materialized); returns (data [64]
        # uint8, scale_bytes [8] uint8). side: 0 = K (linear scale layout),
        # 1 = V (4x4 token-quad swizzle).
        g16 = tl.reshape(xb, (1, 8, 16))
        absx = tl.where(g16 == g16, tl.abs(g16), 0.0)
        amax = tl.max(absx, axis=2)  # [1, 8]
        s = tl.maximum(amax / 6.0, 0.001953125)
        s_code = _e4m3_mag_code(s).to(tl.uint8)  # [1, 8]
        s_val = tl.reshape(_e4m3_decode(s_code), (8,))
        s_vec = tl.reshape(_expand16(tl.reshape(s_val, (1, 8))), (128,))
        # Payload quotient: RZ(x * RN(1/s)) exactly as the reference packer
        # (ref.rz_rounded_product) — NOT the plain RN product.
        q = _rz_product(xb, 1.0 / s_vec)
        code = _e2m1_code(tl.abs(q))
        sign = (tl.cast(q, tl.int32, bitcast=True) < 0).to(tl.int64)
        nib = tl.where(code == 0, 0, code | (sign * 8))
        data = _nibbles_to_bytes(tl.reshape(nib, (1, 128))).to(tl.uint8)
        g = tl.arange(0, 8)
        if side == 0:
            scale_off = tok * 8 + g
        else:
            scale_off = (
                ((tok // 4) * 4 + g // 2) * 8 + (g % 2) * 4 + tok % 4
            )
        scale_bytes = tl.reshape(s_code, (8,))
        return tl.reshape(data, (64,)), scale_bytes, scale_off

    @triton.jit
    def _q4_w1_kernel(
        qkv_ptr,
        q_out_ptr,
        index_q_out_ptr,
        q_norm_w_ptr,
        k_norm_w_ptr,
        iq_norm_w_ptr,
        ik_norm_w_ptr,
        cos_sin_ptr,
        pos_ptr,
        slot_map_ptr,
        index_slot_map_ptr,
        pool_ptr,
        index_cache_ptr,
        eps,
        rotary_dim,
        num_tokens,
        nq,
        nkv,
        niq,
        qkv_row,
        pool_page_stride,
        PROCESS_INDEX: tl.constexpr,
    ):
        pid = tl.program_id(0)
        idx_slots = PROCESS_INDEX * (niq + 1)
        slots_per_token = nq + 2 * nkv + idx_slots
        token = pid // slots_per_token
        slot = pid % slots_per_token
        if token >= num_tokens:
            return
        d = tl.arange(0, 128)
        row_ptr = (
            qkv_ptr + token.to(tl.int64) * qkv_row + slot.to(tl.int64) * 128
        )
        x = tl.load(row_ptr + d).to(tl.float32)
        pos = tl.load(pos_ptr + token)

        is_q = slot < nq
        is_k = (slot >= nq) & (slot < nq + nkv)
        is_v = (slot >= nq + nkv) & (slot < nq + 2 * nkv)

        if is_q:
            xn, r = _gemma_norm(x, q_norm_w_ptr, eps)
            xn = _rope(
                xn, r, row_ptr, q_norm_w_ptr, cos_sin_ptr, pos, rotary_dim
            )
            out = (
                q_out_ptr
                + token.to(tl.int64) * nq * 128
                + slot.to(tl.int64) * 128
            )
            tl.store(out + d, xn.to(tl.bfloat16))
        if is_k:
            xn, r = _gemma_norm(x, k_norm_w_ptr, eps)
            xn = _rope(
                xn, r, row_ptr, k_norm_w_ptr, cos_sin_ptr, pos, rotary_dim
            )
            # The vendor op rewrites the processed row into qkv in place.
            tl.store(row_ptr + d, xn.to(tl.bfloat16))
            sm = tl.load(slot_map_ptr + token)
            if sm >= 0:
                page = sm // 128
                tok = sm % 128
                slot_base = (
                    pool_ptr
                    + page.to(tl.int64) * pool_page_stride
                    + 2 * (slot - nq).to(tl.int64) * 9216
                )
                data, scale_bytes, scale_off = _pack_q4(
                    xn.to(tl.bfloat16).to(tl.float32), tok, 0
                )
                tl.store(slot_base + tok * 64 + tl.arange(0, 64), data)
                tl.store(slot_base + 8192 + scale_off, scale_bytes)
        if is_v:
            sm = tl.load(slot_map_ptr + token)
            if sm >= 0:
                page = sm // 128
                tok = sm % 128
                slot_base = (
                    pool_ptr
                    + page.to(tl.int64) * pool_page_stride
                    + (2 * (slot - nq - nkv) + 1).to(tl.int64) * 9216
                )
                data, scale_bytes, scale_off = _pack_q4(
                    x.to(tl.bfloat16).to(tl.float32), tok, 1
                )
                tl.store(slot_base + tok * 64 + tl.arange(0, 64), data)
                tl.store(slot_base + 8192 + scale_off, scale_bytes)
        if PROCESS_INDEX:
            iq_lo = nq + 2 * nkv
            is_iq = (slot >= iq_lo) & (slot < iq_lo + niq)
            is_ik = slot == iq_lo + niq
            if is_iq:
                xn, r = _gemma_norm(x, iq_norm_w_ptr, eps)
                xn = _rope(
                    xn, r, row_ptr, iq_norm_w_ptr, cos_sin_ptr, pos,
                    rotary_dim,
                )
                # index_q: e4m3(bf16(v)) satfinite bytes. The host passes
                # this buffer as a uint8 byte view of the fp8 e4m3 tensor.
                v_bf = xn.to(tl.bfloat16).to(tl.float32)
                code = _e4m3_mag_code(v_bf)
                sign = (
                    (tl.cast(v_bf, tl.int32, bitcast=True) < 0).to(tl.int64)
                )
                byte = (code | (sign << 7)) & 0xFF
                out = (
                    index_q_out_ptr
                    + token.to(tl.int64) * niq * 128
                    + (slot - iq_lo).to(tl.int64) * 128
                )
                tl.store(out + d, byte.to(tl.uint8))
            if is_ik:
                xn, r = _gemma_norm(x, ik_norm_w_ptr, eps)
                xn = _rope(
                    xn, r, row_ptr, ik_norm_w_ptr, cos_sin_ptr, pos,
                    rotary_dim,
                )
                # In-place rewrite of the processed index_k row, as the
                # vendor op does.
                tl.store(row_ptr + d, xn.to(tl.bfloat16))
                # index_k: e4m3(clamp(v, +-448)) into the fp8 index cache,
                # handed in as a uint8 byte view of the fp8 e4m3 tensor.
                v = tl.minimum(tl.maximum(xn, -448.0), 448.0)
                code = _e4m3_mag_code(v)
                sign = (tl.cast(v, tl.int32, bitcast=True) < 0).to(tl.int64)
                byte = (code | (sign << 7)) & 0xFF
                ism = tl.load(index_slot_map_ptr + token)
                if ism >= 0:
                    tl.store(
                        index_cache_ptr + ism * 128 + d, byte.to(tl.uint8)
                    )

    # ------------------------------------------------------------------
    # R2 (split-K decode reader)
    # ------------------------------------------------------------------

    @triton.jit
    def _dequant_side(
        page_base, slot_idx, n, j, g, pos_mask, swizzled
    ):
        base = page_base + slot_idx * 9216
        u = tl.load(
            base + n[:, None] * 64 + j[None, :],
            mask=pos_mask[:, None],
            other=0,
        )  # [n, 64] uint8
        low = _e2m1_decode(u & 0xF)
        high = _e2m1_decode(u >> 4)
        vals = tl.reshape(tl.join(low, high), (128, 128))  # [n, 128]
        if swizzled:
            sw_t = (n[:, None] // 4) * 4 + g[None, :] // 2
            sw_s = (g[None, :] % 2) * 4 + (n[:, None] % 4)
            sb = tl.load(
                base + 8192 + sw_t * 8 + sw_s,
                mask=pos_mask[:, None],
                other=1,
            )
        else:
            sb = tl.load(
                base + 8192 + n[:, None] * 8 + g[None, :],
                mask=pos_mask[:, None],
                other=1,
            )
        s = _e4m3_decode(sb)  # [n, 8]
        return vals * tl.reshape(_expand16(s), (128, 128))

    @triton.jit
    def _q4_decode_kernel(
        q_ptr,
        pool_ptr,
        t_ptr,
        o_ptr,
        lse_ptr,
        block_table_ptr,
        seq_lens,
        total_q,
        gqa_group_size,
        max_topk,
        sm_scale,
        decode_query_len,
        stride_qn,
        stride_qh,
        stride_qd,
        pool_page_stride,
        stride_th,
        stride_tn,
        stride_tk,
        stride_o_c,
        stride_o_b,
        stride_o_h,
        stride_o_d,
        stride_l_c,
        stride_l_b,
        stride_l_h,
        stride_bt_b,
        NUM_TOPK_CHUNKS: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        sm_scale_log2e = sm_scale * 1.4426950409
        pid_bc = tl.program_id(0)
        pid_kh = tl.program_id(1)
        pid_b = pid_bc % total_q
        pid_c = pid_bc // total_q
        req_id = pid_b // decode_query_len
        q_offset = pid_b - req_id * decode_query_len
        pid_h = pid_kh * gqa_group_size
        chunk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
        chunk_start = pid_c * chunk
        # Loop-bound arithmetic is pinned to int32: sequence lengths fit
        # int32 by construction, and a uniform type keeps the tl.range
        # bounds consistent whether the host tensor is int32 (vLLM) or
        # int64 (the parity harness). Page indices stay int64.
        seq_len = tl.load(seq_lens + req_id).to(tl.int32)
        query_pos = seq_len - decode_query_len + q_offset
        kv_len = tl.maximum(query_pos + 1, 0)
        n_blocks = (kv_len + 127) // 128
        real_topk = tl.minimum(max_topk, n_blocks)
        chunk_end = tl.minimum(chunk_start + chunk, real_topk)

        off_h = tl.arange(0, BLOCK_H)
        h_mask = off_h < gqa_group_size
        d = tl.arange(0, 128)
        n = tl.arange(0, 128)
        j = tl.arange(0, 64)
        g = tl.arange(0, 8)
        q = tl.load(
            q_ptr
            + pid_b * stride_qn
            + (pid_h + off_h)[:, None] * stride_qh
            + d[None, :] * stride_qd,
            mask=h_mask[:, None],
            other=0.0,
        )  # [H, 128]
        m_i = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
        acc = tl.zeros((BLOCK_H, 128), dtype=tl.float32)
        bt_row = block_table_ptr + req_id * stride_bt_b
        cur = t_ptr + pid_kh * stride_th + pid_b * stride_tn
        cur = cur + chunk_start * stride_tk
        for _ in tl.range(chunk_start, chunk_end):
            blk = tl.load(cur).to(tl.int32)
            cur = cur + stride_tk
            valid = (blk >= 0) & (blk < n_blocks)
            safe = tl.where(valid, blk, 0)
            page = tl.load(bt_row + safe).to(tl.int64)
            page_base = pool_ptr + page * pool_page_stride
            pos = safe * 128 + n
            pos_mask = (pos < kv_len) & valid
            kvals = _dequant_side(page_base, pid_kh * 2, n, j, g, pos_mask, 0)
            k = tl.trans(kvals.to(q.dtype))  # [128, n]
            qk = tl.dot(q, k) * sm_scale_log2e
            qk = qk + tl.where(pos_mask[None, :], 0.0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            m_safe = tl.where(valid, m_ij, 0.0)
            p = tl.where(
                pos_mask[None, :], tl.exp2(qk - m_safe[:, None]), 0.0
            )
            l = tl.sum(p, axis=1)
            alpha = tl.where(valid, tl.exp2(m_i - m_safe), 1.0)
            acc = acc * alpha[:, None]
            vvals = _dequant_side(
                page_base, pid_kh * 2 + 1, n, j, g, pos_mask, 1
            )
            v = vvals.to(q.dtype)
            acc = acc + tl.dot(p.to(v.dtype), v)
            lse_new = m_safe + tl.log2(tl.exp2(lse_i - m_safe) + l)
            m_i = tl.where(valid, m_safe, m_i)
            lse_i = tl.where(valid, lse_new, lse_i)
        scale = tl.where(
            lse_i > float("-inf"), tl.exp2(m_i - lse_i), 0.0
        )
        acc = acc * scale[:, None]
        tl.store(
            o_ptr
            + pid_c * stride_o_c
            + pid_b * stride_o_b
            + (pid_h + off_h)[:, None] * stride_o_h
            + d[None, :] * stride_o_d,
            acc.to(o_ptr.dtype.element_ty),
            mask=h_mask[:, None],
        )
        tl.store(
            lse_ptr
            + pid_c * stride_l_c
            + pid_b * stride_l_b
            + (pid_h + off_h) * stride_l_h,
            lse_i,
            mask=h_mask,
        )

    return _q4_w1_kernel, _q4_decode_kernel


_W1_KERNEL = None
_DECODE_KERNEL = None


def _kernels():
    global _W1_KERNEL, _DECODE_KERNEL
    if _W1_KERNEL is None:
        _W1_KERNEL, _DECODE_KERNEL = _build_kernels()
    return _W1_KERNEL, _DECODE_KERNEL


def _dtype_name(t: torch.Tensor) -> str:
    return str(t.dtype).replace("torch.", "")


def _fp8_bytes(t: torch.Tensor | None) -> torch.Tensor | None:
    """Byte view of an fp8 e4m3 buffer (None passes through).

    Convention for every q4 KV Triton kernel: fp8 e4m3 buffers never cross
    the kernel boundary as fp8 pointers. The kernels are pure byte math,
    so the host hands them a uint8 view of the same storage instead; a
    raw fp8 pointer would make Triton's frontend try to cast the stored
    uint8 bytes to fp8e4nv and die at compile time (the GB10 val1
    failure). Validation always runs on the original fp8 tensors before
    the view is taken, so the fail-closed semantics are unchanged.
    """
    if t is None:
        return None
    return t.view(torch.uint8)


# The q4 KV Triton kernels are CUDA-only. A tensor on any other device
# must never reach Triton: it dies there with the opaque "Pointer
# argument cannot be accessed from Triton (cpu tensor?)" of the GB10
# val4 run, which says nothing about WHICH argument was misplaced. The
# wrappers therefore gate placement themselves and name the offender.
# The CPU-only test suite simulates the GPU boundary by patching this
# constant (never by weakening the check itself).
_REQUIRED_DEVICE_TYPE = "cuda"


def _off_device(ref_device: torch.device, named_tensors) -> list:
    """(argument name, device string) for every tensor NOT on ref_device.

    None entries (an argument the caller legitimately omitted, e.g. the
    index buffers on the index-topk reuse layers) pass. Pure function:
    tests call it directly with a CUDA reference device and CPU tensors
    to pin that every kernel-facing argument is named.
    """
    bad = []
    for name, t in named_tensors:
        if t is not None and t.device != ref_device:
            bad.append((name, str(t.device)))
    return bad


def _device_gate(
    query: torch.Tensor, query_name: str, named_tensors
) -> None:
    """Fail closed unless every tensor is on the query's CUDA device."""
    if query.device.type != _REQUIRED_DEVICE_TYPE:
        raise ValueError(
            f"q4 KV refused: '{query_name}' must be on a "
            f"{_REQUIRED_DEVICE_TYPE} device, got {query.device} "
            "(the q4 KV Triton kernels are CUDA-only)"
        )
    bad = _off_device(query.device, named_tensors)
    if bad:
        raise ValueError(
            "q4 KV refused: "
            + "; ".join(
                f"'{name}' is on {dev}, expected {query.device}"
                for name, dev in bad
            )
        )


# ---------------------------------------------------------------------------
# W1 wrapper
# ---------------------------------------------------------------------------


def q4_w1_insert(
    qkv: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_w: torch.Tensor | None,
    index_k_norm_w: torch.Tensor | None,
    num_index_heads: int,
    slot_mapping: torch.Tensor | None,
    index_slot_mapping: torch.Tensor | None,
    pool: torch.Tensor,
    index_cache: torch.Tensor | None,
    block_size: int,
    q_out: torch.Tensor | None,
    index_q_out: torch.Tensor | None,
) -> None:
    """Run W1. Fail closed: refuses unsupported configurations."""
    _device_gate(
        qkv, "qkv",
        (
            ("q_out", q_out),
            ("index_q_out", index_q_out),
            ("q_norm_w", q_norm_w),
            ("k_norm_w", k_norm_w),
            ("index_q_norm_w", index_q_norm_w),
            ("index_k_norm_w", index_k_norm_w),
            ("cos_sin_cache", cos_sin_cache),
            ("positions", positions),
            ("slot_mapping", slot_mapping),
            ("index_slot_mapping", index_slot_mapping),
            ("pool", pool),
            ("index_cache", index_cache),
        ),
    )
    errors = ref.validate_w1_call(
        qkv_dtype=_dtype_name(qkv),
        qkv_width=qkv.shape[1],
        num_tokens=qkv.shape[0],
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_index_heads=num_index_heads,
        rotary_dim=rotary_dim,
        block_size=block_size,
        pool_dtype=_dtype_name(pool),
        pool_dim=pool.dim(),
        pool_slots=pool.shape[1],
        pool_tokens=pool.shape[2],
        pool_width=pool.shape[3],
        pool_row_stride=pool.stride(2),
        pool_slot_stride=pool.stride(1),
        q_out_present=q_out is not None,
        q_fp8_out_present=False,
        index_cache_dtype=(
            _dtype_name(index_cache) if index_cache is not None else None
        ),
        index_q_out_dtype=(
            _dtype_name(index_q_out) if index_q_out is not None else None
        ),
        kv_k_scale=1.0,
        kv_v_scale=1.0,
    )
    if errors:
        raise ValueError("q4 W1 refused: " + "; ".join(errors))
    process_index = num_index_heads > 0
    slots_per_token = num_heads + 2 * num_kv_heads
    if process_index:
        slots_per_token += num_index_heads + 1
    grid = (qkv.shape[0] * slots_per_token,)
    w1, _ = _kernels()
    w1[grid](
        qkv,
        q_out,
        # fp8 e4m3 buffers cross the boundary as uint8 byte views
        # (_fp8_bytes); the kernel writes raw e4m3 bytes, never fp8 values.
        _fp8_bytes(index_q_out) if index_q_out is not None else qkv,
        q_norm_w,
        k_norm_w,
        index_q_norm_w if index_q_norm_w is not None else q_norm_w,
        index_k_norm_w if index_k_norm_w is not None else q_norm_w,
        cos_sin_cache,
        positions,
        slot_mapping if slot_mapping is not None else positions,
        index_slot_mapping if index_slot_mapping is not None else positions,
        pool,
        _fp8_bytes(index_cache) if index_cache is not None else pool,
        eps,
        rotary_dim,
        qkv.shape[0],
        num_heads,
        num_kv_heads,
        num_index_heads,
        qkv.shape[1],
        pool.stride(0),
        PROCESS_INDEX=process_index,
    )


# ---------------------------------------------------------------------------
# R2 wrapper (split-K partials + vLLM's unchanged merge kernel)
# ---------------------------------------------------------------------------


def _next_pow2(x: int) -> int:
    return 1 << (max(1, x) - 1).bit_length()


def q4_sparse_decode(
    q: torch.Tensor,  # [total_q, H, 128]
    pool: torch.Tensor,  # uint8 [pages, 2*nkv, 128, 72]
    topk_idx: torch.Tensor,  # [nkv, total_q, topk]
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, H, 128]
    decode_query_len: int,
) -> None:
    """Run R2 + the unchanged vLLM merge kernel. Fail closed on geometry."""
    _device_gate(
        q, "q",
        (
            ("pool", pool),
            ("topk_idx", topk_idx),
            ("block_table", block_table),
            ("seq_lens", seq_lens),
            ("output", output),
        ),
    )
    total_q, num_heads, head_dim = q.shape
    if head_dim != 128:
        raise ValueError("q4 decode requires head_dim == 128")
    if pool.dtype is not torch.uint8:
        raise ValueError(
            "q4 decode requires the pool as raw uint8 bytes, got "
            f"{_dtype_name(pool)}; the overlay must re-view the container "
            "before dispatch"
        )
    if pool.dim() != 4 or pool.shape[1] != 2 * num_kv_heads:
        raise ValueError("q4 pool geometry mismatch")
    if pool.stride(2) != 72 or pool.stride(1) != 9216 or pool.stride(3) != 1:
        raise ValueError("q4 pool slot regions must be contiguous")
    assert total_q == seq_lens.shape[0] * decode_query_len
    max_topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    # Shape-constant chunk count (cuda-graph safe), as in the vendor wrapper.
    target = max(1, min(max_topk, 256 // max(1, total_q * num_kv_heads)))
    num_topk_chunks = 1 << (target.bit_length() - 1)
    o_partial = torch.empty(
        num_topk_chunks, total_q, num_heads, head_dim,
        dtype=q.dtype, device=q.device,
    )
    lse_partial = torch.empty(
        num_topk_chunks, total_q, num_heads,
        dtype=torch.float32, device=q.device,
    )
    _, decode_kernel = _kernels()
    grid = (total_q * num_topk_chunks, num_kv_heads)
    decode_kernel[grid](
        q,
        pool,
        topk_idx,
        o_partial,
        lse_partial,
        block_table,
        seq_lens,
        total_q,
        gqa_group_size,
        max_topk,
        sm_scale,
        decode_query_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        pool.stride(0),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        block_table.stride(0),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        BLOCK_H=16 if gqa_group_size <= 16 else _next_pow2(gqa_group_size),
    )
    # Merge with vLLM's existing kernel (unchanged; sm_121 has no PDL).
    from vllm.models.minimax_m3.common.ops.sparse_attn import (
        _merge_topk_attn_out_kernel,
    )

    merge_grid = (total_q, num_heads)
    _merge_topk_attn_out_kernel[merge_grid](
        o_partial,
        lse_partial,
        output,
        head_dim,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_PDL=False,
    )
