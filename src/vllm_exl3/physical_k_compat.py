"""Derive fused routed-expert K from the loaded trellis, not config defaults.

Mixed-K packs are self-describing: ``LinearEXL3.K`` comes from the physical
``trellis.shape[-1] // 16``. A layer may be physically uniform at K5/K7 while
its config/base ``bits`` is different. Fused kernels must receive the physical
uniform K or they will interpret the trellis with the wrong width.
"""
from __future__ import annotations

from typing import Any


def _physical_k_values(inners: list[dict[str, Any]]) -> set[int]:
    values: set[int] = set()
    for pack in inners:
        gateless = pack.get("gate") is None
        for projection in ("gate", "up", "down"):
            if gateless and projection == "gate":
                # Non-gated MoE (NemotronH): no gate projection exists; the
                # fused state aliases up pointer tables into the gate slots.
                continue
            linear = pack[projection]
            k = getattr(linear, "K", None)
            if k is None:
                trellis = getattr(linear, "trellis", None)
                if trellis is None or getattr(trellis, "ndim", 0) != 3:
                    raise RuntimeError(
                        f"EXL3 fused state cannot determine physical K for {projection}"
                    )
                words = int(trellis.shape[-1])
                if words <= 0 or words % 16:
                    raise RuntimeError(
                        f"EXL3 fused state invalid trellis width {words} for {projection}"
                    )
                k = words // 16
            values.add(int(k))
    return values


def install_physical_fused_k_compat(exl3_module: Any) -> None:
    original = getattr(exl3_module, "build_exl3_fused_state", None)
    if not callable(original):
        return
    if bool(getattr(original, "_vllm_exl3_physical_k_wrapped", False)):
        return

    def wrapped(layer: Any, inners: list[dict[str, Any]]) -> None:
        physical = _physical_k_values(inners)
        # [LAB PATCH 2026-10-06] The uniformity requirement exists because the
        # scalar fused launches interpret every trellis with one width. The
        # native multi-K kernel (p2b_fused_moe_mk) reads per-expert K tables
        # from the device, so a physically mixed layer is legitimate whenever
        # that kernel is compiled in.
        mixed_native = len(physical) != 1 and bool(
            getattr(
                exl3_module, "_native_moe_mixed_kernel_available", lambda: False
            )()
        )
        if len(physical) != 1 and not mixed_native:
            raise RuntimeError(
                "EXL3 fused MoE requires one physical K across gate/up/down and "
                f"all local experts; loaded K values={sorted(physical)}"
            )
        physical_k = next(iter(physical)) if len(physical) == 1 else max(physical)
        if not 1 <= physical_k <= 8:
            raise RuntimeError(
                f"EXL3 fused MoE physical K={physical_k} is outside K1-K8"
            )

        original(layer, inners)
        if mixed_native and getattr(layer, "_exl3_mixed_native_mk", False):
            return
        configured = int(getattr(layer, "_exl3_bits", physical_k))
        layer._exl3_k = physical_k
        layer._exl3_physical_fused_k = physical_k
        layer._exl3_configured_k = configured

        if configured != physical_k:
            logger = getattr(exl3_module, "logger", None)
            if logger is not None:
                getattr(logger, "info_once", logger.info)(
                    "EXL3 fused layer physical K overrides config/base K: "
                    "physical=%s configured=%s",
                    physical_k,
                    configured,
                )

    wrapped._vllm_exl3_physical_k_wrapped = True
    wrapped._vllm_exl3_original = original
    exl3_module.build_exl3_fused_state = wrapped
    exl3_module._vllm_exl3_physical_fused_k_compat_installed = True
