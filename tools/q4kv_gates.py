"""Numerics and byte gates for the q4 KV parity harness (pure torch, CPU-safe).

Why the W1 gates look like this (the GB10 val3 lesson): the Triton W1
kernel and the pure-torch reference implement the same algorithm, but
two fp32 details are deliberately different —

* the kernel's reciprocal square root is the hardware ``sqrt.approx.f32``
  (as the vendor op's ``rsqrtf``), while the reference uses the
  precisely-rounded 1/sqrt;
* the norm's sum-of-squares reduction order differs (Triton's tree
  reduction vs torch's reduction order).

Both effects are bounded: a handful of fp32 ulps, ~2^-21 relative. A
value perturbed by 2^-21 relative either rounds to the same bf16 number
or to the adjacent one, so the bf16 outputs of the two paths differ by
at most 1 bf16 ulp, on a small fraction of elements. Bit-exact
comparison of the post-norm+RoPE rows is therefore NOT a meaningful gate
(the GB10 val3 run failed exactly there, on expected noise); what
matters is:

(a) **norm+RoPE numerics gate** — the kernel's post-norm+RoPE rows (the
    bf16 ``q_out`` buffer and the processed K / index_k rows the kernel
    writes back into ``qkv`` in place) must stay within
    ``BF16_ULP_LIMIT`` bf16 ulp of the reference everywhere, with
    cosine >= ``COS_LIMIT`` against it. A wrong RoPE pairing, norm
    epsilon, weight offset or reduction length produces multi-ulp local
    damage and/or collapses the cosine; the 2-ulp limit carries a full
    2x margin over the <= 1 ulp the two legitimate noise sources can
    produce, and a uniform 1-ulp shift (cos ~ 1 - 2^-17) still passes
    while a uniform 2-ulp shift (cos ~ 1 - 2^-15 < 0.99999) fails.
(b) **packing byte gate, decoupled from the norm rounding** — the
    reference packer run on the kernel's OWN processed rows must
    reproduce the kernel's pool bytes byte-for-byte. Both packers see
    bit-identical fp32 inputs there (the kernel packs the bf16-rounded
    row it also writes back, and bf16-rounding is idempotent), so every
    differing byte is a real packing bug (nibble pairing, scale layout,
    encoder, scatter offset). The fp8 index outputs cannot be gated this
    way — the vendor op and the reference both encode the fp32
    in-register value, which never exists in bf16 form to read back —
    so index_q/index_k bytes get the e4m3 code-distance gate
    (``E4M3_CODE_LIMIT`` code steps + cosine) instead: the fp32 noise
    can move a byte by at most 1 code step, with probability ~2^-18 per
    element.

All gate functions return a diagnostics dict with a ``passed`` flag and
never raise on comparison data (shape/dtype mismatches fail with a
``reason`` instead). They are exercised on synthetic perturbations by
``tools/gb10_q4kv_parity.py --cpu-dry-run`` and pinned by the CPU tests.
"""

from __future__ import annotations

import torch

from vllm_exl3 import minimax_q4kv_reference as ref

# <= 2 bf16 ulp of local damage anywhere (expected noise: <= 1 ulp).
BF16_ULP_LIMIT = 2
# Overall shape agreement (expected noise: cos >= 1 - 2^-30).
COS_LIMIT = 0.99999
# <= 2 e4m3 code steps for the fp8 index bytes (expected noise: <= 1 step
# on a ~2^-18 fraction of codes).
E4M3_CODE_LIMIT = 2


def _monotone_key(bits: torch.Tensor, sign_bit: int, mask: int) -> torch.Tensor:
    """Map sign-magnitude float bit patterns to a monotone integer key.

    Positive patterns keep their order shifted above the negative range;
    negative patterns are mirrored so that adjacent representable values
    (in either direction) differ by exactly 1.
    """
    sign = 1 << sign_bit
    return torch.where(bits < sign, bits + sign, mask - bits)


def _bits_from_key(key: torch.Tensor, sign_bit: int, mask: int) -> torch.Tensor:
    """Inverse of _monotone_key."""
    sign = 1 << sign_bit
    return torch.where(key >= sign, key - sign, mask - key)


def bf16_ulp_step(
    t: torch.Tensor, step: int, *, fraction: float, seed: int
) -> torch.Tensor:
    """Synthetic-perturbation helper (dry run / tests): move a random
    ``fraction`` of the FINITE elements of a bf16 tensor by exactly
    ``step`` representable values in value order (step > 0 = away from
    -inf). Steps that would enter a non-finite pattern (inf/NaN) are
    skipped."""
    flat = t.reshape(-1).clone()
    bits = flat.view(torch.int16).to(torch.int32) & 0xFFFF
    key = _monotone_key(bits, 15, 0xFFFF)
    n = int(flat.numel())
    k = max(1, int(n * fraction))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:k]
    new_key = (key[idx] + step).clamp(0, 0xFFFF)
    new_bits = _bits_from_key(new_key, 15, 0xFFFF)
    # Only step finite elements, and never into inf/NaN (the exponent
    # 0xFF patterns 0x7F80-0x7FFF / 0xFF80-0xFFFF).
    ok = torch.isfinite(flat[idx]) & ((new_bits & 0x7F80) != 0x7F80)
    bits[idx[ok]] = new_bits[ok]
    return bits.to(torch.int16).view(torch.bfloat16).reshape(t.shape)


def e4m3_code_step(
    t: torch.Tensor, step: int, *, fraction: float, seed: int
) -> torch.Tensor:
    """Synthetic-perturbation helper (dry run / tests): move a random
    ``fraction`` of the elements of a uint8 e4m3 byte tensor by exactly
    ``step`` representable-value steps (sign-magnitude code steps, as
    measured by gate_e4m3_codes). Steps that would enter the NaN code
    (0x7F/0xFF) are skipped."""
    flat = t.reshape(-1).clone().to(torch.int32) & 0xFF
    key = _monotone_key(flat, 7, 0xFF)
    n = int(flat.numel())
    k = max(1, int(n * fraction))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:k]
    new_key = (key[idx] + step).clamp(0, 0xFF)
    new_bits = _bits_from_key(new_key, 7, 0xFF)
    new_bits = torch.where(
        (new_bits & 0x7F) == 0x7F, flat[idx], new_bits
    )
    flat[idx] = new_bits
    return flat.to(torch.uint8).reshape(t.shape)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).to(torch.float64)
    b = b.reshape(-1).to(torch.float64)
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float((a @ b) / denom)


def _base_diag(gate: str, label: str) -> dict:
    return {"gate": gate, "label": label}


def gate_bf16_numerics(
    candidate: torch.Tensor, reference: torch.Tensor, label: str
) -> dict:
    """Gate (a) for bf16 rows: max ulp distance <= BF16_ULP_LIMIT and
    cosine >= COS_LIMIT; full metrics (max_abs, max ulp, mismatched
    fraction, cosine) are reported either way."""
    diag = _base_diag("bf16-ulp", label)
    if (
        candidate.shape != reference.shape
        or candidate.dtype != reference.dtype
        or candidate.dtype != torch.bfloat16
    ):
        return {
            **diag,
            "passed": False,
            "reason": (
                f"expected equal-shape bfloat16 tensors, got "
                f"{tuple(candidate.shape)} {candidate.dtype} vs "
                f"{tuple(reference.shape)} {reference.dtype}"
            ),
        }
    c = candidate.reshape(-1)
    r = reference.reshape(-1)
    cb = c.view(torch.int16).to(torch.int32) & 0xFFFF
    rb = r.view(torch.int16).to(torch.int32) & 0xFFFF
    ck = _monotone_key(cb, 15, 0xFFFF)
    rk = _monotone_key(rb, 15, 0xFFFF)
    ulp = (ck - rk).abs()
    both_finite = torch.isfinite(c) & torch.isfinite(r)
    bit_equal = cb == rb
    # Finiteness must agree; two non-finite values must be bit-identical.
    finiteness_violation = int(
        ((~torch.isfinite(c)) != (~torch.isfinite(r))).sum()
    ) + int(((~both_finite) & ~bit_equal).sum())
    mismatch = ~bit_equal & both_finite
    n = int(c.numel())
    n_mismatch = int(mismatch.sum())
    if both_finite.any():
        max_ulp = int(ulp[both_finite].max())
        max_abs = float(
            (c[mismatch].to(torch.float32) - r[mismatch].to(torch.float32))
            .abs()
            .max()
        ) if n_mismatch else 0.0
        cos = _cosine(c[both_finite], r[both_finite])
    else:
        max_ulp, max_abs, cos = 1 << 30, 0.0, 0.0
    passed = (
        finiteness_violation == 0
        and max_ulp <= BF16_ULP_LIMIT
        and cos >= COS_LIMIT
    )
    reason = None
    if finiteness_violation:
        reason = f"{finiteness_violation} finiteness mismatches"
    elif max_ulp > BF16_ULP_LIMIT:
        reason = f"max ulp distance {max_ulp} > {BF16_ULP_LIMIT}"
    elif cos < COS_LIMIT:
        reason = f"cosine {cos:.8f} < {COS_LIMIT}"
    return {
        **diag,
        "passed": passed,
        "reason": reason,
        "n_elements": n,
        "n_mismatch": n_mismatch,
        "mismatch_fraction": n_mismatch / n,
        "max_abs": max_abs,
        "max_ulp": max_ulp,
        "cosine": cos,
    }


def gate_e4m3_codes(
    candidate: torch.Tensor, reference: torch.Tensor, label: str
) -> dict:
    """Gate for fp8 e4m3 byte outputs (index_q bytes, index_k scatter
    bytes): max code distance <= E4M3_CODE_LIMIT (in representable-value
    steps of the magnitude code, sign included via the monotone key) and
    cosine >= COS_LIMIT of the decoded values."""
    diag = _base_diag("e4m3-code", label)
    if (
        candidate.shape != reference.shape
        or candidate.dtype != reference.dtype
        or candidate.dtype != torch.uint8
    ):
        return {
            **diag,
            "passed": False,
            "reason": (
                f"expected equal-shape uint8 byte tensors, got "
                f"{tuple(candidate.shape)} {candidate.dtype} vs "
                f"{tuple(reference.shape)} {reference.dtype}"
            ),
        }
    cb = candidate.reshape(-1).to(torch.int32) & 0xFF
    rb = reference.reshape(-1).to(torch.int32) & 0xFF
    ck = _monotone_key(cb, 7, 0xFF)
    rk = _monotone_key(rb, 7, 0xFF)
    dist = (ck - rk).abs()
    bit_equal = cb == rb
    # The NaN code (0x7F/0xFF) must not appear on one side only.
    nan_violation = int(
        (((cb & 0x7F) == 0x7F) != ((rb & 0x7F) == 0x7F)).sum()
    ) + int((~bit_equal & ((cb & 0x7F) == 0x7F) & ((rb & 0x7F) == 0x7F)).sum())
    mismatch = ~bit_equal
    n = int(cb.numel())
    n_mismatch = int(mismatch.sum())
    max_dist = int(dist.max()) if n else 0
    cv = ref.e4m3_decode(candidate)
    rv = ref.e4m3_decode(reference)
    both_finite = torch.isfinite(cv) & torch.isfinite(rv)
    if both_finite.any():
        cos = _cosine(cv[both_finite], rv[both_finite])
    else:
        cos = 0.0
    passed = (
        nan_violation == 0 and max_dist <= E4M3_CODE_LIMIT and cos >= COS_LIMIT
    )
    reason = None
    if nan_violation:
        reason = f"{nan_violation} NaN-code mismatches"
    elif max_dist > E4M3_CODE_LIMIT:
        reason = f"max code distance {max_dist} > {E4M3_CODE_LIMIT}"
    elif cos < COS_LIMIT:
        reason = f"decoded cosine {cos:.8f} < {COS_LIMIT}"
    return {
        **diag,
        "passed": passed,
        "reason": reason,
        "n_elements": n,
        "n_mismatch": n_mismatch,
        "mismatch_fraction": n_mismatch / n,
        "max_code_distance": max_dist,
        "decoded_cosine": cos,
    }


def gate_bytes_exact(
    candidate: torch.Tensor, reference: torch.Tensor, label: str
) -> dict:
    """Gate (b): byte-exact comparison (pool data nibbles, e4m3 scale
    bytes, scatter targets). Any differing byte fails the gate; the first
    mismatching flat offset is reported for triage."""
    diag = _base_diag("bytes-exact", label)
    if (
        candidate.shape != reference.shape
        or candidate.dtype != reference.dtype
        or candidate.dtype != torch.uint8
    ):
        return {
            **diag,
            "passed": False,
            "reason": (
                f"expected equal-shape uint8 byte tensors, got "
                f"{tuple(candidate.shape)} {candidate.dtype} vs "
                f"{tuple(reference.shape)} {reference.dtype}"
            ),
        }
    c = candidate.reshape(-1)
    r = reference.reshape(-1)
    mismatch = c != r
    n = int(c.numel())
    n_mismatch = int(mismatch.sum())
    first = int(mismatch.nonzero()[0, 0]) if n_mismatch else -1
    return {
        **diag,
        "passed": n_mismatch == 0,
        "reason": None if n_mismatch == 0 else (
            f"{n_mismatch} differing bytes, first at flat offset {first}"
        ),
        "n_elements": n,
        "n_mismatch": n_mismatch,
        "mismatch_fraction": n_mismatch / n,
        "first_mismatch_offset": first,
    }
