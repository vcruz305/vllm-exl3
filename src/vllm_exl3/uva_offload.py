"""Experimental vLLM UVA placement guard for packed EXL3 routed experts.

Current vLLM can place selected parameters in pinned host memory and expose them
as accelerator views through Unified Virtual Addressing (UVA). CUDA kernels can
then dereference those mapped pointers without first making the whole parameter
resident in VRAM.

This module does not claim performance or hardware qualification. It only makes
the placement contract explicit and fail-closed when a recipe requests it.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

UVA_REQUIRED_ENV = "VLLM_EXL3_REQUIRE_UVA_EXPERTS"

# The large per-layer packed payloads created by Exl3MoEMethod. Codebook marker
# tensors are tiny and intentionally remain outside the required placement set.
EXL3_MOE_UVA_PARAMETER_SEGMENTS: tuple[str, ...] = (
    "w13_trellis",
    "w13_suh",
    "w13_svh",
    "w2_trellis",
    "w2_suh",
    "w2_svh",
)


@dataclass(frozen=True)
class UvaExpertLayerStatus:
    applicable: bool
    required: bool
    fully_uva_offloaded: bool
    partially_uva_offloaded: bool
    cpu_fallback_parameters: tuple[str, ...]
    uva_parameters: tuple[str, ...]
    resident_parameters: tuple[str, ...]
    parameter_segments: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def uva_expert_offload_required() -> bool:
    value = os.environ.get(UVA_REQUIRED_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _device_type(param: Any) -> str:
    device = getattr(param, "device", None)
    return str(getattr(device, "type", device or "unknown"))


def inspect_exl3_moe_uva_layer(
    layer: Any,
    *,
    required: bool | None = None,
) -> UvaExpertLayerStatus:
    """Inspect vLLM's UVA marker/device state on one EXL3 RoutedExperts layer."""
    if required is None:
        required = uva_expert_offload_required()

    if not hasattr(layer, "w13_trellis"):
        return UvaExpertLayerStatus(
            applicable=False,
            required=bool(required),
            fully_uva_offloaded=False,
            partially_uva_offloaded=False,
            cpu_fallback_parameters=(),
            uva_parameters=(),
            resident_parameters=(),
            parameter_segments=EXL3_MOE_UVA_PARAMETER_SEGMENTS,
        )

    uva: list[str] = []
    cpu: list[str] = []
    resident: list[str] = []
    missing: list[str] = []

    for name in EXL3_MOE_UVA_PARAMETER_SEGMENTS:
        param = getattr(layer, name, None)
        if param is None:
            missing.append(name)
            continue
        marked = bool(getattr(param, "_vllm_is_uva_offloaded", False))
        device_type = _device_type(param)
        if marked:
            uva.append(name)
        elif device_type == "cpu":
            cpu.append(name)
        else:
            resident.append(name)

    if missing:
        raise RuntimeError(
            "EXL3 UVA placement inspection found an incomplete routed-expert "
            f"payload: missing {missing}"
        )

    fully = len(uva) == len(EXL3_MOE_UVA_PARAMETER_SEGMENTS)
    partial = bool(uva) and not fully
    return UvaExpertLayerStatus(
        applicable=True,
        required=bool(required),
        fully_uva_offloaded=fully,
        partially_uva_offloaded=partial,
        cpu_fallback_parameters=tuple(cpu),
        uva_parameters=tuple(uva),
        resident_parameters=tuple(resident),
        parameter_segments=EXL3_MOE_UVA_PARAMETER_SEGMENTS,
    )


def validate_exl3_moe_uva_layer(layer: Any) -> UvaExpertLayerStatus:
    """Require a complete mapped-UVA expert payload for an experimental run.

    vLLM's non-UVA fallback keeps parameters as ordinary CPU tensors and moves
    state during a wrapped module forward. The EXL3 routed-expert quant method
    launches its own packed kernels, so that fallback is not accepted here.
    """
    status = inspect_exl3_moe_uva_layer(layer, required=True)
    if not status.applicable:
        return status
    if status.cpu_fallback_parameters:
        raise RuntimeError(
            "EXL3 expert UVA was requested, but vLLM left packed parameters as "
            "ordinary CPU tensors instead of mapped accelerator views: "
            f"{list(status.cpu_fallback_parameters)}. Verify CUDA UVA support "
            "and that VLLM_WEIGHT_OFFLOADING_DISABLE_UVA is not set."
        )
    if status.partially_uva_offloaded:
        raise RuntimeError(
            "EXL3 expert UVA offload is partial; refusing mixed placement for "
            f"the qualification path. UVA={list(status.uva_parameters)} "
            f"resident={list(status.resident_parameters)}"
        )
    if not status.fully_uva_offloaded:
        raise RuntimeError(
            "EXL3 expert UVA was required but none of the packed expert payload "
            "parameters carry vLLM's UVA-offload marker. Configure vLLM "
            "--offload-backend uva, a sufficient --cpu-offload-gb budget, and "
            "--cpu-offload-params with the EXL3 expert parameter segments."
        )
    return status


def install_uva_expert_validation(exl3_module: Any) -> None:
    """Wrap Exl3MoEMethod post-load processing with an opt-in UVA guard."""
    method_cls = getattr(exl3_module, "Exl3MoEMethod", None)
    if method_cls is None:
        return
    original = getattr(method_cls, "process_weights_after_loading", None)
    if not callable(original) or bool(
        getattr(original, "_vllm_exl3_uva_guard_wrapped", False)
    ):
        return

    def process_weights_after_loading_uva_guard(self, layer):
        if uva_expert_offload_required() and hasattr(layer, "w13_trellis"):
            status = validate_exl3_moe_uva_layer(layer)
            layer._exl3_uva_expert_status = status.to_dict()
        return original(self, layer)

    process_weights_after_loading_uva_guard._vllm_exl3_uva_guard_wrapped = True
    setattr(
        method_cls,
        "process_weights_after_loading",
        process_weights_after_loading_uva_guard,
    )
