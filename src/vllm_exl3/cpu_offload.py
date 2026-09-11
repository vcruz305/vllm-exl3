"""CPU-offload compatibility planning for EXL3 routed experts.

This module is intentionally policy-only. ``vllm-exl3`` does not currently
provide a host-resident CPU expert executor. The helper captures the eligibility
contract of upstream ExLlamaV3's experimental CPU-MoE path so recipes can fail
closed instead of implying support from the presence of EXL3 weights alone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

DEEPSEEK_V41_ARCH = "DeepseekV41ForCausalLM"
EXLLAMAV3_CPU_MOE_CODEBOOK = "mul1"
EXLLAMAV3_CPU_MOE_MAX_K = 8


@dataclass(frozen=True)
class CpuOffloadPlan:
    architecture: str
    codebooks: tuple[str, ...]
    max_k: int | None
    uniform_expert_biases: bool | None
    exllamav3_architecture_available: bool | None
    exllamav3_metadata_candidate: bool
    vllm_exl3_execution_available: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    recommended_route: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize_codebooks(codebooks: str | Iterable[str] | None) -> tuple[str, ...]:
    if codebooks is None:
        return ()
    if isinstance(codebooks, str):
        values = [codebooks]
    else:
        values = list(codebooks)
    normalized = {
        str(value).strip().lower()
        for value in values
        if str(value).strip()
    }
    return tuple(sorted(normalized))


def plan_exllamav3_cpu_offload(
    *,
    architecture: str,
    codebooks: str | Iterable[str] | None,
    max_k: int | None,
    uniform_expert_biases: bool | None = None,
    exllamav3_architecture_available: bool | None = None,
) -> CpuOffloadPlan:
    """Return a fail-closed plan for upstream ExLlamaV3 CPU-MoE.

    Current upstream CPU-MoE requires mul1 experts, K <= 8 and uniform expert
    bias presence. For DeepSeek-V4.1, standalone inference additionally requires
    a forward-correct ``DeepseekV41ForCausalLM`` architecture in ExLlamaV3.

    ``vllm-exl3`` itself has no CPU expert executor today; this helper never
    changes that fact and never enables a backend.
    """
    architecture = str(architecture).strip()
    normalized = _normalize_codebooks(codebooks)
    blockers: list[str] = []
    warnings: list[str] = []

    if not architecture:
        blockers.append("checkpoint architecture is unknown")

    if exllamav3_architecture_available is False:
        blockers.append(
            f"ExLlamaV3 cannot instantiate architecture {architecture!r}"
        )
    elif exllamav3_architecture_available is None:
        warnings.append(
            "ExLlamaV3 architecture availability was not proven"
        )

    if not normalized:
        blockers.append("EXL3 codebook was not proven")
    elif normalized != (EXLLAMAV3_CPU_MOE_CODEBOOK,):
        blockers.append(
            "ExLlamaV3 CPU-MoE currently requires mul1-only experts; "
            f"observed {list(normalized)!r}"
        )

    if max_k is None:
        warnings.append(
            f"maximum K was not proven; CPU-MoE requires K <= {EXLLAMAV3_CPU_MOE_MAX_K}"
        )
    else:
        if isinstance(max_k, bool) or not isinstance(max_k, int) or max_k <= 0:
            blockers.append("max_k must be a positive integer when provided")
        elif max_k > EXLLAMAV3_CPU_MOE_MAX_K:
            blockers.append(
                f"K={max_k} exceeds ExLlamaV3 CPU-MoE limit "
                f"{EXLLAMAV3_CPU_MOE_MAX_K}"
            )

    if uniform_expert_biases is False:
        blockers.append(
            "expert bias presence is not uniform, which CPU-MoE requires"
        )
    elif uniform_expert_biases is None:
        warnings.append("uniform expert-bias presence was not proven")

    metadata_candidate = not blockers
    if architecture == DEEPSEEK_V41_ARCH and exllamav3_architecture_available is not True:
        route = (
            "complete and qualify the standalone DeepSeek-V4.1 ExLlamaV3 forward "
            "port before attempting CPU-MoE"
        )
    elif normalized != (EXLLAMAV3_CPU_MOE_CODEBOOK,):
        route = (
            "produce/qualify a mul1 sibling pack or add and qualify an upstream "
            "CPU backend for the observed codebook"
        )
    elif metadata_candidate:
        route = (
            "run ExLlamaV3 CPU-MoE as an external runtime experiment and compare "
            "against the source model; vllm-exl3 is not the CPU executor"
        )
    else:
        route = "resolve the blockers before runtime qualification"

    return CpuOffloadPlan(
        architecture=architecture,
        codebooks=normalized,
        max_k=max_k,
        uniform_expert_biases=uniform_expert_biases,
        exllamav3_architecture_available=exllamav3_architecture_available,
        exllamav3_metadata_candidate=metadata_candidate,
        vllm_exl3_execution_available=False,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        recommended_route=route,
    )
