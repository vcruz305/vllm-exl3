"""Opt-in lossless tensor-granular routed EXL3 storage.

Whole experts only. Payloads are registered at their exact checkpoint shapes as
weights arrive, rather than allocating a layer-wide K or a max-K padded bank.
The zero-sized w13_/w2_ parameters are vLLM loader endpoints, NOT kernel storage.
No source-format metadata is inferred here. vLLM owns the model and collectives.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.nn import Parameter

PROJECTIONS = {"w1": "gate", "w3": "up", "w2": "down"}


def tensor_mixed_k_enabled() -> bool:
    return os.environ.get("VLLM_EXL3_TENSOR_MIXED_K", "0") == "1"


@dataclass(frozen=True)
class ExpertLayout:
    """Launch compatibility key; eligibility is NOT hardware qualification."""

    bits: tuple[int, int, int]
    codebooks: tuple[str, str, str]
    hidden: int
    intermediate: int

    @property
    def fused_eligible(self) -> bool:
        from .deepseek_v41 import exllamav3_fused_geometry_supported

        return (
            len(self.bits) == len(self.codebooks) == 3
            and all(type(k) is int and 2 <= k <= 8 for k in self.bits)
            and len(set(self.codebooks)) == 1
            and self.codebooks[0] in ("mcg", "mul1")
            and exllamav3_fused_geometry_supported(self.hidden, self.intermediate)
        )

    @property
    def native_eligible(self) -> bool:
        return (
            self.fused_eligible
            and self.codebooks == ("mcg",) * 3
            and len(set(self.bits)) == 1
            and self.bits[0] in (2, 3, 4)
        )


def group_layouts(layouts):
    """Stable local->bucket membership; empty banks produce no launch groups."""
    groups = {}
    for expert, layout in enumerate(layouts):
        groups.setdefault(layout, []).append(expert)
    return {layout: tuple(members) for layout, members in groups.items()}


class TensorMixedKStore(torch.nn.Module):
    """Exact, independently owned registered tensors, indexed by LOCAL expert."""

    def __init__(self, num_experts: int, hidden: int, intermediate: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden = hidden
        self.intermediate = intermediate
        self.payload = torch.nn.ParameterDict()
        self.sealed = False

    def check_expert_map(self, current_map):
        old_map = self.expert_map_at_load
        if (old_map is None) != (current_map is None) or (
            old_map is not None and not torch.equal(old_map, current_map)
        ):
            raise ValueError(
                "EXL3 mixed-K expert map changed; dynamic rebalancing is unsupported"
            )

    def tensor(self, expert: int, projection: str, suffix: str):
        return self.payload[f"e{expert}_{projection}_{suffix}"]

    def load(self, expert: int, shard: str, suffix: str, source: torch.Tensor, device):
        from .exl3 import EXL3_SUFFIXES

        if self.sealed:
            raise ValueError(
                "EXL3 mixed-K storage is sealed; reloading/rebalancing is unsupported"
            )
        if (
            not 0 <= expert < self.num_experts
            or shard not in PROJECTIONS
            or suffix not in EXL3_SUFFIXES
        ):
            raise ValueError(
                f"EXL3 mixed-K invalid load target: {expert}/{shard}/{suffix}"
            )
        projection = PROJECTIONS[shard]
        key = f"e{expert}_{projection}_{suffix}"
        if key in self.payload:
            raise ValueError(f"EXL3 mixed-K duplicate load: {key}")
        in_dim, out_dim = (
            (self.intermediate, self.hidden)
            if shard == "w2"
            else (self.hidden, self.intermediate)
        )
        if suffix == "trellis":
            valid = (
                source.dtype == torch.int16
                and source.ndim == 3
                and tuple(source.shape[:2]) == (in_dim // 16, out_dim // 16)
                and source.shape[-1] % 16 == 0
                and 2 <= source.shape[-1] // 16 <= 8
            )
        elif suffix in ("suh", "svh"):
            valid = source.dtype == torch.float16 and tuple(source.shape) == (
                in_dim if suffix == "suh" else out_dim,
            )
        else:
            valid = source.dtype == torch.int32 and tuple(source.shape) in ((), (1,))
        if not valid:
            raise ValueError(
                f"EXL3 mixed-K invalid {key}: shape={tuple(source.shape)} dtype={source.dtype}; "
                f"expected exact {in_dim}x{out_dim} geometry, K2-K8 int16 trellis, "
                "fp16 scale vectors or one int32 marker; no padding/casting allowed"
            )
        value = torch.empty_like(
            source, device=device, memory_format=torch.contiguous_format
        )
        value.copy_(source.detach())
        self.payload[key] = Parameter(value, requires_grad=False)

    def build_inners(self, factory):
        from .exl3 import _check_moe_codebook_markers

        # Check the entire local bank before constructing any native handle.
        records = []
        for expert in range(self.num_experts):
            packs = {}
            for projection in PROJECTIONS.values():
                prefix = f"e{expert}_{projection}"
                required = [f"{prefix}_{s}" for s in ("trellis", "suh", "svh")]
                markers = {
                    s: self.payload.get(f"{prefix}_{s}") for s in ("mcg", "mul1")
                }
                if any(k not in self.payload for k in required) or all(
                    v is None for v in markers.values()
                ):
                    raise ValueError(f"EXL3 mixed-K incomplete payload: {prefix}")
                if all(v is not None for v in markers.values()):
                    raise ValueError(
                        f"EXL3 mixed-K codebook conflict: {prefix} has both markers"
                    )
                present = next(v for v in markers.values() if v is not None)
                zero = torch.zeros_like(present)
                try:
                    _check_moe_codebook_markers(
                        markers["mcg"] if markers["mcg"] is not None else zero,
                        markers["mul1"] if markers["mul1"] is not None else zero,
                        prefix,
                    )
                except RuntimeError as exc:
                    raise ValueError(
                        f"EXL3 mixed-K invalid codebook {prefix}: {exc}"
                    ) from exc
                packs[projection] = [self.payload[k] for k in required] + [
                    markers["mcg"],
                    markers["mul1"],
                ]
            records.append(packs)
        layouts = [
            ExpertLayout(
                tuple(pack[p][0].shape[-1] // 16 for p in PROJECTIONS.values()),
                tuple(
                    "mcg" if pack[p][3] is not None else "mul1"
                    for p in PROJECTIONS.values()
                ),
                self.hidden,
                self.intermediate,
            )
            for pack in records
        ]
        inners = [
            {p: factory(*values) for p, values in pack.items()} for pack in records
        ]
        self.groups = group_layouts(layouts)
        self.sealed = True
        return inners


def validate_runtime_config(config):
    """Fail before allocation for engine modes not covered by this reference path."""
    if not getattr(getattr(config, "model_config", None), "enforce_eager", False):
        raise ValueError(
            "EXL3 mixed-K reference requires --enforce-eager (vLLM remains graph owner)"
        )
    if getattr(getattr(config, "parallel_config", None), "enable_eplb", False):
        raise ValueError(
            "EXL3 mixed-K reference requires static expert placement; disable EPLB"
        )
    if getattr(config, "speculative_config", None) is not None:
        raise ValueError(
            "EXL3 mixed-K reference has no DSpark/speculative qualification; disable speculation"
        )
    offload = getattr(config, "offload_config", None)
    if (
        getattr(getattr(offload, "uva", None), "cpu_offload_gb", 0) > 0
        or getattr(getattr(offload, "prefetch", None), "offload_group_size", 0) > 0
        or getattr(getattr(config, "cache_config", None), "cpu_offload_gb", 0) > 0
    ):
        raise ValueError(
            "EXL3 mixed-K lazy registered storage does not support weight offload/UVA"
        )


def create_mixed_weights(method, layer, num_experts, hidden, intermediate, extra):
    from .exl3 import (
        _VLLM_AVAILABLE,
        EXL3_SUFFIXES,
        _exl3_routed_experts_loader,
        _resolve_tp_geometry,
        set_weight_attrs,
    )

    if _VLLM_AVAILABLE:
        from vllm.config import get_current_vllm_config

        validate_runtime_config(get_current_vllm_config())
    if _resolve_tp_geometry(layer) != (0, 1):
        raise ValueError(
            "EXL3 mixed-K reference currently requires whole experts (MoE TP=1; use EP)"
        )
    layer._exl3_mixed_store = TensorMixedKStore(num_experts, hidden, intermediate)
    emap = getattr(layer, "expert_map", None)
    layer._exl3_mixed_store.register_buffer(
        "expert_map_at_load",
        None if emap is None else emap.detach().clone(),
        persistent=False,
    )
    for prefix in ("w13", "w2"):
        for suffix in EXL3_SUFFIXES:
            dtype = (
                torch.int16
                if suffix == "trellis"
                else (torch.int32 if suffix in ("mcg", "mul1") else torch.float16)
            )
            param = Parameter(torch.empty(0, dtype=dtype), requires_grad=False)
            layer.register_parameter(f"{prefix}_{suffix}", param)
            set_weight_attrs(
                param, {k: v for k, v in extra.items() if k != "weight_loader"}
            )
            param.weight_loader = method._load_exl3
            param._exl3_owner = layer
    layer._exl3_hidden_size = hidden
    layer._exl3_intermediate_local = intermediate
    layer.load_weights = _exl3_routed_experts_loader(layer)


def require_eager_execution(*, compiling: bool, capturing: bool):
    if compiling or capturing:
        raise ValueError(
            "EXL3 mixed-K reference is eager-only; vLLM must disable compilation/CUDA graphs"
        )


def apply_mixed_reference(x, ids, weights, layer, *, limit=None):
    from .exl3 import apply_exl3_python_loop, pin_exl3_expert_map

    require_eager_execution(
        compiling=torch.compiler.is_compiling(),
        capturing=x.is_cuda and torch.cuda.is_current_stream_capturing(),
    )
    store = layer._exl3_mixed_store
    if not store.sealed:
        raise ValueError("EXL3 mixed-K incomplete/unsealed storage")
    if (
        x.ndim != 2
        or x.shape[1] != store.hidden
        or ids.ndim != 2
        or ids.shape[0] != x.shape[0]
        or ids.shape != weights.shape
        or ids.dtype not in (torch.int32, torch.int64)
        or not x.is_floating_point()
        or not weights.is_floating_point()
        or ids.device != x.device
        or weights.device != x.device
    ):
        raise ValueError(
            "EXL3 mixed-K requires [tokens,hidden] input and matching integer IDs/float weights"
        )
    store.check_expert_map(getattr(layer, "expert_map", None))
    result = apply_exl3_python_loop(
        x,
        ids.to(torch.long),
        weights,
        layer._exl3_inners,
        pin_exl3_expert_map(layer, x.device),
        limit,
    )
    layer._exl3_last_apply = "mixed_k_reference"
    return result.to(x.dtype)
