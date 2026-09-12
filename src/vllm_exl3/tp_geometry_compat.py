"""Compatibility shim for vLLM RoutedExperts TP/EP geometry.

Current vLLM stores the authoritative per-MoE tensor-parallel geometry under
``RoutedExperts.moe_config.moe_parallel_config``.  Expert parallelism may set
that MoE TP size to 1 even while the process-wide tensor-parallel world is >1.
Falling back to the process TP group in that case incorrectly slices whole EP
experts as if they were TP-sharded.
"""
from __future__ import annotations

from typing import Any


def nested_moe_tp_geometry(*owners: Any) -> tuple[int, int] | None:
    """Return ``(tp_rank, tp_size)`` from current vLLM nested MoE config."""
    for owner in owners:
        if owner is None:
            continue

        # RoutedExperts -> FusedMoEConfig -> FusedMoEParallelConfig.
        moe_config = getattr(owner, "moe_config", None)
        if moe_config is None and hasattr(owner, "moe_parallel_config"):
            moe_config = owner
        if moe_config is None:
            continue

        parallel = getattr(moe_config, "moe_parallel_config", None)
        if parallel is None:
            continue
        rank = getattr(parallel, "tp_rank", None)
        size = getattr(parallel, "tp_size", None)
        if rank is None and size is None:
            continue

        resolved_rank = int(rank) if rank is not None else 0
        resolved_size = int(size) if size is not None else 1
        if resolved_size < 1 or resolved_rank < 0 or resolved_rank >= resolved_size:
            raise ValueError(
                "invalid vLLM MoE TP geometry: "
                f"tp_rank={resolved_rank} tp_size={resolved_size}"
            )
        return resolved_rank, resolved_size
    return None


def install_tp_geometry_compat(exl3_module: Any) -> None:
    """Wrap ``exl3._resolve_tp_geometry`` with nested-current-vLLM support."""
    if bool(getattr(exl3_module, "_vllm_exl3_tp_geometry_compat_installed", False)):
        return

    original = exl3_module._resolve_tp_geometry

    def resolve(*owners: Any) -> tuple[int, int]:
        nested = nested_moe_tp_geometry(*owners)
        if nested is not None:
            return nested
        return original(*owners)

    resolve._vllm_exl3_nested_tp_wrapped = True  # type: ignore[attr-defined]
    exl3_module._resolve_tp_geometry = resolve
    exl3_module._vllm_exl3_tp_geometry_compat_installed = True
