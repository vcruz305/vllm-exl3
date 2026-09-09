"""Runtime policy helpers for EXL3 serving.

This module is an independent vllm-exl3 implementation built around this
project's own GB10 measurements and runtime interfaces. Recent public work in
MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks reinforced the usefulness of
workload-aware dispatch and explicit effective-configuration reporting. No
post-relicense source code from that repository is copied here.
See THIRD_PARTY_NOTICES.md and docs/provenance.md.
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


def fused_temp_rows_requested(default: int = 2048) -> int:
    """Return the requested fused scratch row count without applying it.

    The existing extension/runtime contract was built around the historical
    value.  Until smaller arenas are GPU-qualified end to end, diagnostics may
    expose a requested value but serving must keep the actual constant intact.
    """
    return _positive_int_env("VLLM_EXL3_FUSED_TEMP_ROWS", default)


def native_row_cap(bits: int, fallback: Callable[[int], int]) -> int:
    """Resolve the effective native decode-row ceiling for one bit width."""
    key = f"VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K{int(bits)}"
    raw = os.environ.get(key)
    if raw is not None:
        try:
            return max(0, int(raw.strip(), 10))
        except (AttributeError, TypeError, ValueError):
            pass
    return max(0, int(fallback(int(bits))))


def install_native_row_policy(exl3_module: Any) -> bool:
    """Make per-bit row ceilings part of the real EXL3 dispatch resolver.

    This wraps, rather than replaces, the plugin's existing resolver. Global
    overrides and the measured-cap policy therefore remain the fallback when a
    per-bit variable is absent. Repeated plugin registration is idempotent.
    """
    if getattr(exl3_module, "_vllm_exl3_per_bit_policy_installed", False):
        return False
    original = getattr(exl3_module, "_native_moe_max_rows", None)
    if not callable(original):
        raise RuntimeError("vllm-exl3 native row-cap resolver is unavailable")

    def effective(bits: int) -> int:
        return native_row_cap(int(bits), original)

    effective.__name__ = getattr(original, "__name__", "_native_moe_max_rows")
    effective.__doc__ = "Effective native row cap with optional per-bit override."
    exl3_module._vllm_exl3_native_row_cap_base = original
    exl3_module._native_moe_max_rows = effective
    exl3_module._vllm_exl3_per_bit_policy_installed = True
    return True


def diagnostics(
    *,
    backend: str,
    native_available: bool,
    native_abi: int,
    native_caps: dict[int, int],
    fused_rows_actual: int,
    fused_rows_requested: int,
    fat_threshold: int,
    fat_kernel_available: bool,
    spec_schedule: str,
    per_bit_policy_installed: bool,
) -> dict[str, Any]:
    """Build a stable, JSON-friendly effective-policy record."""
    return {
        "moe_backend": backend,
        "native_available": bool(native_available),
        "native_abi": int(native_abi),
        "native_row_caps": {str(k): int(v) for k, v in sorted(native_caps.items())},
        "per_bit_native_policy_installed": bool(per_bit_policy_installed),
        "fused_temp_rows_actual": int(fused_rows_actual),
        "fused_temp_rows_requested": int(fused_rows_requested),
        "fused_temp_rows_override_active": False,
        "fat_expert_threshold": int(fat_threshold),
        "fat_kernel_available": bool(fat_kernel_available),
        "spec_schedule": spec_schedule,
    }
