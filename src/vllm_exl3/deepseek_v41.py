"""DeepSeek V4.1 compatibility helpers for the EXL3 serving plugin.

This module intentionally does not reimplement the DeepSeek V4.1 model. vLLM owns
that architecture. The helpers here bridge EXL3's mixed-checkpoint metadata into
vLLM's model-level quantization probes and describe the TP4+EP4 layout that is a
better fit for EXL3 on four DGX Sparks.

The policy is independently implemented from public vLLM/DeepSeek interfaces.
No DeepSeek or vLLM source code is copied into this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import re
from typing import Any, Mapping

DEEPSEEK_V41_MAIN_EXPERTS = 384
DEEPSEEK_V41_DSPARK_EXPERTS = 128
DEEPSEEK_V41_TOPK = 6
DEEPSEEK_V41_HIDDEN = 5120
DEEPSEEK_V41_INTERMEDIATE = 2304
DEEPSEEK_V41_DSPARK_STAGES = 3
DEEPSEEK_V41_DSPARK_TOKENS = 5
V41_NATIVE_MOE_ENV = "VLLM_EXL3_V41_NATIVE_MOE"
V41_NATIVE_MOE_ABI = 3


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalize_block_size(value: object) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        rows, cols = int(value[0]), int(value[1])
    except (TypeError, ValueError, OverflowError):
        return None
    if rows <= 0 or cols <= 0:
        return None
    return rows, cols


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def source_quantization_config(config: object) -> Mapping[str, Any]:
    """Return the declared non-routed/source quantization metadata."""
    return _mapping(getattr(config, "non_routed_quantization", None))


def source_weight_block_size(config: object) -> list[int] | None:
    """Expose the source dense-weight block shape to architecture-level probes.

    DeepSeek V4.1's vLLM model inspects the *global* quantization config before
    individual dense layers ask EXL3 for their delegated quant method. EXL3 packs
    that preserve the source MXFP8 dense weights therefore need to surface the
    delegate's [32, 32] block shape at the outer config as well.
    """
    block = _normalize_block_size(source_quantization_config(config).get("weight_block_size"))
    return list(block) if block is not None else None


def is_deepseek_v41_source_quant(config: object) -> bool:
    """Whether config declares the native V4/V4.1 MXFP8 dense-weight delegate."""
    source = source_quantization_config(config)
    method = str(source.get("quant_method", "")).strip().lower()
    return method == "deepseek_v4_fp8" and _normalize_block_size(
        source.get("weight_block_size")
    ) == (32, 32)


def _layer_index(prefix: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(prefix))
    return int(match.group(1)) if match else None


def should_delegate_dspark_source(
    config: object,
    *,
    prefix: str,
    global_num_experts: int | None,
) -> bool:
    """Return True when a V4.1 DSpark routed block should retain source MXFP4.

    Existing packs can continue to declare ``mtp_experts_start_layer``. For V4.1
    packs that omit it, the 128-expert DSpark blocks are distinguishable from the
    384-expert main stack, so the plugin can safely infer the source delegation
    when the V4.1 dense-source quantization signature is also present.
    """
    if str(getattr(config, "mtp_experts", "exl3")).strip().lower() != "source":
        return False

    layer_index = _layer_index(prefix)
    explicit_start = getattr(config, "mtp_experts_start_layer", None)
    if explicit_start is not None:
        try:
            return layer_index is not None and layer_index >= int(explicit_start)
        except (TypeError, ValueError, OverflowError):
            return False

    return (
        is_deepseek_v41_source_quant(config)
        and global_num_experts == DEEPSEEK_V41_DSPARK_EXPERTS
    )


def native_p2b_geometry_supported(hidden_size: int, intermediate_size: int) -> bool:
    """Whether ABI-3 p2b geometry can cover the dimensions without 128 tails."""
    return (
        hidden_size > 0
        and intermediate_size > 0
        and hidden_size % 128 == 0
        and intermediate_size % 128 == 0
    )


def exllamav3_fused_geometry_supported(hidden_size: int, intermediate_size: int) -> bool:
    """Whether the layer geometry is suitable for ExLlamaV3's fused MoE path.

    Expert-count itself is not capped at 128. The historical ``>128`` fallback
    in vllm-exl3 refers to *tokens routed to one expert* in a batch, not the
    number of experts owned by the layer. ExLlamaV3's fused kernel accepts a
    num-experts-sized pointer table and has compiled K1-K8 instances.
    """
    return native_p2b_geometry_supported(hidden_size, intermediate_size)


def v41_native_moe_requested() -> bool:
    """V4.1 native p2b remains opt-in until GB10 parity/throughput qualification."""
    return _env_enabled(V41_NATIVE_MOE_ENV, False)


@dataclass(frozen=True)
class DeepseekV41Plan:
    tensor_parallel_size: int
    expert_parallel: bool
    moe_tensor_parallel_size: int
    expert_parallel_size: int
    global_experts: int
    local_experts: int
    hidden_size: int
    intermediate_size: int
    intermediate_size_per_rank: int
    top_k: int
    exllamav3_fused_candidate: bool
    native_p2b_candidate: bool
    native_p2b_requires_abi: int
    native_p2b_default_enabled: bool
    preferred_first_boot_backend: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def plan_deepseek_v41(
    *,
    tensor_parallel_size: int = 4,
    expert_parallel: bool = True,
    global_experts: int = DEEPSEEK_V41_MAIN_EXPERTS,
    hidden_size: int = DEEPSEEK_V41_HIDDEN,
    intermediate_size: int = DEEPSEEK_V41_INTERMEDIATE,
    top_k: int = DEEPSEEK_V41_TOPK,
) -> DeepseekV41Plan:
    """Describe the routed-expert layout for a DeepSeek V4.1 deployment.

    Under vLLM expert parallelism, the MoE stops tensor-sharding each expert and
    instead shards whole experts over the original TP group. TP4 owns 96 complete
    5120x2304 experts per rank; TP2 owns 192 complete experts per rank. Both EP
    layouts keep the full 2304 intermediate width and avoid pure-TP4's 576 tail.
    """
    values = {
        "tensor_parallel_size": tensor_parallel_size,
        "global_experts": global_experts,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "top_k": top_k,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if global_experts % tensor_parallel_size:
        raise ValueError("global_experts must divide evenly across TP/EP ranks")
    if not expert_parallel and intermediate_size % tensor_parallel_size:
        raise ValueError("intermediate_size must divide evenly for pure TP")

    if expert_parallel:
        moe_tp = 1
        ep_size = tensor_parallel_size
        local_experts = global_experts // ep_size
        inter_local = intermediate_size
    else:
        moe_tp = tensor_parallel_size
        ep_size = 1
        local_experts = global_experts
        inter_local = intermediate_size // tensor_parallel_size

    exllamav3_candidate = exllamav3_fused_geometry_supported(hidden_size, inter_local)
    native_candidate = native_p2b_geometry_supported(hidden_size, inter_local)

    if expert_parallel and exllamav3_candidate:
        backend = "exllamav3"
        reason = (
            f"TP{tensor_parallel_size}+EP{tensor_parallel_size} keeps {local_experts} "
            f"full experts per rank at {hidden_size}x{inter_local}. ExLlamaV3 is "
            "the correctness-first backend; ABI-3 native p2b is an explicit A/B "
            "for K2-K4 layers only."
        )
    elif exllamav3_candidate:
        backend = "exllamav3"
        reason = "layout is fused-compatible, but expert parallel is preferred on Spark"
    else:
        backend = "loop"
        reason = "layout has a 128-wide geometry tail and requires a fallback path"

    return DeepseekV41Plan(
        tensor_parallel_size=tensor_parallel_size,
        expert_parallel=expert_parallel,
        moe_tensor_parallel_size=moe_tp,
        expert_parallel_size=ep_size,
        global_experts=global_experts,
        local_experts=local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        intermediate_size_per_rank=inter_local,
        top_k=top_k,
        exllamav3_fused_candidate=exllamav3_candidate,
        native_p2b_candidate=native_candidate,
        native_p2b_requires_abi=V41_NATIVE_MOE_ABI,
        native_p2b_default_enabled=False,
        preferred_first_boot_backend=backend,
        reason=reason,
    )


def _install_native_geometry_wrapper(exl3_module: object) -> None:
    original = getattr(exl3_module, "_native_moe_dimensions_supported", None)
    if not callable(original) or bool(
        getattr(original, "_vllm_exl3_v41_geometry_wrapped", False)
    ):
        return

    def dimensions_supported_v41(x2d, layer, inners, limit=None):
        if original(x2d, layer, inners, limit):
            return True
        if not v41_native_moe_requested():
            return False
        try:
            if x2d.dim() != 2 or not x2d.is_cuda:
                return False
            if limit is not None and (not math.isfinite(limit) or limit < 0):
                return False
            hidden = int(getattr(layer, "_exl3_hidden_size", x2d.shape[1]))
            intermediate = int(getattr(layer, "_exl3_intermediate_local", 0))
            bits = int(getattr(layer, "_exl3_k", getattr(layer, "_exl3_bits", -1)))
            rows = int(x2d.shape[0])
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False

        if not (
            hidden == DEEPSEEK_V41_HIDDEN
            and intermediate == DEEPSEEK_V41_INTERMEDIATE
            and int(x2d.shape[1]) == hidden
            and native_p2b_geometry_supported(hidden, intermediate)
            and bits in (2, 3, 4)
            and rows >= 1
            and len(inners) > 0
        ):
            return False

        native = exl3_module._load_native_exl3_ext()
        if native is None or int(getattr(native, "P2B_MOE_ABI_VERSION", 0)) < V41_NATIVE_MOE_ABI:
            return False
        return rows <= exl3_module._native_moe_max_rows(bits)

    dimensions_supported_v41._vllm_exl3_v41_geometry_wrapped = True  # type: ignore[attr-defined]
    setattr(exl3_module, "_native_moe_dimensions_supported", dimensions_supported_v41)


def install_deepseek_v41_compat(exl3_module: object) -> None:
    """Install narrow V4.1 compatibility shims onto ``Exl3Config`` once.

    The shim has three jobs only:
    1. Surface the delegated source block shape through the outer EXL3 config so
       vLLM V4.1 can choose the correct MXFP8 scale naming/layout.
    2. Allow V4.1's 128-expert DSpark blocks to remain source MXFP4 when a pack
       requests ``mtp_experts=source`` but omits a numeric start-layer marker.
    3. Add an opt-in ABI-3 native geometry path for the aligned V4.1 EP expert shape.

    All actual model architecture, attention, Engram, routing and DSpark execution
    remain owned by vLLM.
    """
    if bool(getattr(exl3_module, "_vllm_exl3_v41_compat_installed", False)):
        return

    config_cls = getattr(exl3_module, "Exl3Config")

    if "weight_block_size" not in config_cls.__dict__:
        setattr(config_cls, "weight_block_size", property(source_weight_block_size))

    if "has_blocked_weights" not in config_cls.__dict__:
        def has_blocked_weights(self) -> bool:
            return source_weight_block_size(self) is not None
        setattr(config_cls, "has_blocked_weights", has_blocked_weights)

    original_get_quant_method = config_cls.get_quant_method
    if not bool(getattr(original_get_quant_method, "_vllm_exl3_v41_wrapped", False)):
        def get_quant_method_v41(self, layer, prefix):
            try:
                from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
            except Exception:
                RoutedExperts = ()  # type: ignore[assignment]

            if RoutedExperts and isinstance(layer, RoutedExperts):
                global_num_experts = getattr(layer, "global_num_experts", None)
                if global_num_experts is None:
                    global_num_experts = getattr(
                        getattr(layer, "moe_config", None), "num_experts", None
                    )
                if should_delegate_dspark_source(
                    self,
                    prefix=prefix,
                    global_num_experts=global_num_experts,
                ):
                    resolver = getattr(self, "_mtp_expert_method", None)
                    if callable(resolver):
                        method, _how = resolver(layer, prefix)
                        if method is not None:
                            return method
            return original_get_quant_method(self, layer, prefix)

        get_quant_method_v41._vllm_exl3_v41_wrapped = True  # type: ignore[attr-defined]
        setattr(config_cls, "get_quant_method", get_quant_method_v41)

    _install_native_geometry_wrapper(exl3_module)
    setattr(exl3_module, "_vllm_exl3_v41_compat_installed", True)
