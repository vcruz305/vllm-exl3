"""Planning policy for future grouped routed-expert prefill execution.

The performance motivation is informed by public grouped-expert work, including
MiaAI-Lab's GLM-5.3-Flash E3 results. This module is independently implemented
around vllm-exl3's own K2/K3 serving constraints. It does not reproduce their
CUDA kernels, routing-table builder, or launcher code. See docs/provenance.md.

The policy deliberately does *not* dispatch a kernel. It defines a bounded,
fail-closed contract that a GPU implementation can consume after qualification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from typing import Iterable

GROUPED_PREFILL_ENV = "VLLM_EXL3_GROUPED_PREFILL"
GROUPED_PREFILL_MAX_ROWS_ENV = "VLLM_EXL3_GROUPED_PREFILL_MAX_ROWS"
DEFAULT_GROUPED_MAX_ROWS = 8192
DEFAULT_TILE_ROWS = 64
DEFAULT_SUPPORTED_BITS = (2, 3)


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    return default


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def grouped_prefill_enabled() -> bool:
    """Whether the experimental grouped-prefill policy was requested."""
    return _env_enabled(GROUPED_PREFILL_ENV, False)


def grouped_prefill_max_rows(default: int = DEFAULT_GROUPED_MAX_ROWS) -> int:
    """Maximum routed rows held in one grouped working-set window."""
    if default <= 0:
        raise ValueError("default must be positive")
    return _positive_int_env(GROUPED_PREFILL_MAX_ROWS_ENV, default)


def grouped_prefill_scratch_bytes(
    rows: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype_bytes: int = 2,
) -> int:
    """Conservative scratch budget for one grouped-prefill row window.

    The budget assumes transient fp16-class storage for a gathered hidden row,
    gate+up intermediates, and activated/down-input values. A concrete fused
    kernel may require less; this is an admission ceiling, not a live allocation.
    """
    for name, value in {
        "rows": rows,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "dtype_bytes": dtype_bytes,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if rows and (hidden_size == 0 or intermediate_size == 0 or dtype_bytes == 0):
        raise ValueError("non-empty scratch requires positive dimensions and dtype_bytes")
    return rows * (hidden_size + 3 * intermediate_size) * dtype_bytes


@dataclass(frozen=True)
class GroupedPrefillPlan:
    requested: bool
    eligible: bool
    reason: str
    bits: int
    codebook: str
    routed_rows: int
    fat_threshold: int
    tile_rows: int
    window_rows: int
    windows: int
    scratch_bytes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def plan_grouped_prefill(
    *,
    bits: int,
    codebook: str,
    has_mul1: bool,
    hidden_size: int,
    intermediate_size: int,
    routed_rows: int,
    fat_threshold: int = 256,
    tile_rows: int = DEFAULT_TILE_ROWS,
    max_rows: int | None = None,
    requested: bool | None = None,
    supported_bits: Iterable[int] = DEFAULT_SUPPORTED_BITS,
) -> GroupedPrefillPlan:
    """Return a bounded grouped-prefill candidate plan.

    Eligibility covers local format/shape invariants only. It does not infer GPU
    architecture support and never implies that a grouped executor is installed.
    """
    ints = {
        "bits": bits,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "routed_rows": routed_rows,
        "fat_threshold": fat_threshold,
        "tile_rows": tile_rows,
    }
    for name, value in ints.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if bits <= 0 or hidden_size <= 0 or intermediate_size <= 0 or routed_rows < 0:
        raise ValueError("bits/dimensions must be positive and routed_rows non-negative")
    if fat_threshold < 0 or tile_rows <= 0:
        raise ValueError("fat_threshold must be non-negative and tile_rows positive")

    requested = grouped_prefill_enabled() if requested is None else bool(requested)
    supported = tuple(int(v) for v in supported_bits)
    limit = grouped_prefill_max_rows() if max_rows is None else max_rows
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("max_rows must be a positive integer")
    if limit < tile_rows:
        raise ValueError("max_rows must fit at least one tile")

    reason = "eligible"
    eligible = True
    normalized_codebook = str(codebook).strip().lower()
    if not requested:
        eligible, reason = False, "disabled"
    elif bits not in supported:
        eligible, reason = False, f"unsupported_bits:{bits}"
    elif normalized_codebook != "mcg":
        eligible, reason = False, f"unsupported_codebook:{codebook}"
    elif has_mul1:
        eligible, reason = False, "mul1_not_supported"
    elif hidden_size % 256:
        eligible, reason = False, "hidden_not_multiple_of_256"
    elif intermediate_size % 128:
        eligible, reason = False, "intermediate_not_multiple_of_128"
    elif routed_rows <= fat_threshold:
        eligible, reason = False, "below_fat_threshold"

    window_rows = 0
    windows = 0
    scratch = 0
    if eligible:
        aligned_limit = (limit // tile_rows) * tile_rows
        required = math.ceil(routed_rows / tile_rows) * tile_rows
        window_rows = min(required, aligned_limit)
        windows = math.ceil(routed_rows / window_rows)
        scratch = grouped_prefill_scratch_bytes(window_rows, hidden_size, intermediate_size)

    return GroupedPrefillPlan(
        requested=requested,
        eligible=eligible,
        reason=reason,
        bits=bits,
        codebook=normalized_codebook,
        routed_rows=routed_rows,
        fat_threshold=fat_threshold,
        tile_rows=tile_rows,
        window_rows=window_rows,
        windows=windows,
        scratch_bytes=scratch,
    )
