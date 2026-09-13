"""Safety guards for mixed-K trellis prescan.

The low-memory arena prescan maps local expert ids onto checkpoint expert ids.
That mapping is only valid when vLLM uses linear expert placement. Other
placement/EPLB strategies must fall back to the normal loader, which already
uses vLLM's authoritative expert map.
"""
from __future__ import annotations

from typing import Any


def _placement_strategy(layer: Any) -> str | None:
    value = getattr(layer, "expert_placement_strategy", None)
    if value is None:
        manager = getattr(layer, "expert_map_manager", None)
        value = getattr(manager, "placement_strategy", None) if manager is not None else None
    return str(value).lower() if value is not None else None


def _eplb_active(layer: Any) -> bool:
    for owner in (layer, getattr(layer, "expert_map_manager", None)):
        if owner is None:
            continue
        for name in ("enable_eplb", "eplb_enabled"):
            if bool(getattr(owner, name, False)):
                return True
    return False


def install_mixed_k_prescan_guard(exl3_module: Any) -> None:
    current = getattr(exl3_module, "_try_prescan_trellis_shapes", None)
    if not callable(current):
        return
    if bool(getattr(current, "_vllm_exl3_mixed_k_prescan_guard", False)):
        return

    def guarded(layer: Any, num_experts: int):
        placement = _placement_strategy(layer)
        if _eplb_active(layer) or placement not in (None, "linear"):
            logger = getattr(exl3_module, "logger", None)
            if logger is not None:
                getattr(logger, "info_once", logger.info)(
                    "EXL3 mixed-K arena prescan disabled for expert placement=%s "
                    "eplb=%s; falling back to authoritative loader mapping",
                    placement or "unknown",
                    _eplb_active(layer),
                )
            return None
        return current(layer, num_experts)

    guarded._vllm_exl3_mixed_k_prescan_guard = True
    guarded._vllm_exl3_original = current
    exl3_module._try_prescan_trellis_shapes = guarded
    exl3_module._vllm_exl3_mixed_k_prescan_guard_installed = True
