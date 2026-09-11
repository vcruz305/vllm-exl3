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
import re
from typing import Any, Mapping

DEEPSEEK_V41_MAIN_EXPERTS = 384
DEEPSEEK_V41_DSPARK_EXPERTS = 128
DEEPSEEK_V41_TOPK = 6
DEEPSEEK_V41_HIDDEN = 5120
DEEPSEEK_V41_INTERMEDIATE = 2304
DEEPSEEK_V41_DSPARK_STAGES = 3
DEEPSEEK_V41_DSPARK_TOKENS = 5
EXLLAMAV3_FUSED_EXPERT_LIMIT = 128


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
    instead shards whole experts over the original TP group. This is the desired
    four-Spark EXL3 layout: 96 complete 5120x2304 experts per rank rather than
    384 experts with an awkward 576-wide TP partition.
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

    exllamav3_candidate = (
        local_experts <= EXLLAMAV3_FUSED_EXPERT_LIMIT
        and hidden_size % 128 == 0
        and inter_local % 128 == 0
    )
    # Current vllm-exl3 native p2b ABI is still the qualified 4096 x {1024,2048}
    # family. Keep this false for V4.1 until the dedicated SM121 specialization
    # is implemented and GPU-qualified.
    native_candidate = hidden_size == 4096 and inter_local in (1024, 2048)

    if expert_parallel and exllamav3_candidate:
        backend = "exllamav3"
        reason = (
            "TP4+EP4 keeps full experts on each rank: 96 local experts at "
            "5120x2304, within the ExLlamaV3 fused expert-count/alignment envelope"
        )
    elif exllamav3_candidate:
        backend = "exllamav3"
        reason = "layout is fused-compatible, but pure TP is not the preferred Spark path"
    else:
        backend = "loop"
        reason = "layout exceeds the currently qualified fused EXL3 envelope"

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
        preferred_first_boot_backend=backend,
        reason=reason,
    )


def install_deepseek_v41_compat(exl3_module: object) -> None:
    """Install narrow V4.1 compatibility shims onto ``Exl3Config`` once.

    The shim has two jobs only:
    1. Surface the delegated source block shape through the outer EXL3 config so
       vLLM V4.1 can choose the correct MXFP8 scale naming/layout.
    2. Allow V4.1's 128-expert DSpark blocks to remain source MXFP4 when a pack
       requests ``mtp_experts=source`` but omits a numeric start-layer marker.

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

    setattr(exl3_module, "_vllm_exl3_v41_compat_installed", True)
