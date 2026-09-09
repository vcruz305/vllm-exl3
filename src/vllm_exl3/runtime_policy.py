"""Runtime policy helpers for EXL3 serving.

This module is an independent vllm-exl3 implementation built around this
project's own GB10 measurements and runtime interfaces. Recent public work in
MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks reinforced the usefulness of
workload-aware verification/dispatch and explicit effective-configuration
reporting. No post-relicense source code from that repository is copied here.
See THIRD_PARTY_NOTICES.md for the earlier MIT-derived code already present in
this project and docs/provenance.md for design provenance.
"""

from __future__ import annotations

import os
from typing import Any, Callable


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip(), 10)
    except (AttributeError, TypeError, ValueError):
        return default
    return value if value > 0 else default


def fused_temp_rows(default: int = 2048) -> int:
    """Resident fused-MoE row capacity requested by this process.

    This is opt-in and defaults to the historical 2048-row contract. A caller
    may reduce the value only after proving its runner never sends larger fused
    calls. The dispatch path should fail safely rather than overrun the arena.
    """
    return _positive_int_env("VLLM_EXL3_FUSED_TEMP_ROWS", default)


def native_row_cap(bits: int, fallback: Callable[[int], int]) -> int:
    """Resolve a per-bit native decode-row ceiling.

    Per-bit overrides let TP1 users A/B K2 separately from K3/K4 instead of
    applying one global row cap to different trellis costs.
    """
    key = f"VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K{int(bits)}"
    raw = os.environ.get(key)
    if raw is not None:
        try:
            return max(0, int(raw.strip(), 10))
        except (AttributeError, TypeError, ValueError):
            pass
    return max(0, int(fallback(int(bits))))


def diagnostics(
    *,
    backend: str,
    native_available: bool,
    native_abi: int,
    native_caps: dict[int, int],
    fused_rows: int,
    fat_threshold: int,
    fat_kernel_available: bool,
    spec_schedule: str,
) -> dict[str, Any]:
    """Build a stable, JSON-friendly effective-policy record."""
    return {
        "moe_backend": backend,
        "native_available": bool(native_available),
        "native_abi": int(native_abi),
        "native_row_caps": {str(k): int(v) for k, v in sorted(native_caps.items())},
        "fused_temp_rows": int(fused_rows),
        "fat_expert_threshold": int(fat_threshold),
        "fat_kernel_available": bool(fat_kernel_available),
        "spec_schedule": spec_schedule,
    }
