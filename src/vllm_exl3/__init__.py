"""Register the routed-expert EXL3 implementation with a local vLLM runtime."""

__all__ = [
    "register",
    "get_speculative_draft_tokens",
    "parse_speculative_schedule",
    "is_adaptive_verification_enabled",
    "filter_speculative_candidates",
    "compute_mla_kv_cache_bytes",
    "validate_context_scaling",
    "runtime_diagnostics",
    "GroupedPrefillPlan",
    "grouped_prefill_enabled",
    "grouped_prefill_max_rows",
    "grouped_prefill_scratch_bytes",
    "plan_grouped_prefill",
    "DeepseekV41Plan",
    "plan_deepseek_v41",
    "source_weight_block_size",
    "is_deepseek_v41_source_quant",
    "should_delegate_dspark_source",
    "CpuOffloadPlan",
    "plan_exllamav3_cpu_offload",
    "UvaExpertLayerStatus",
    "inspect_exl3_moe_uva_layer",
    "validate_exl3_moe_uva_layer",
    "uva_expert_offload_required",
    "supported_config_bits",
    "native_qualified_bits",
]


def register() -> None:
    # Importing the module executes its register_quantization_config decorator.
    from . import exl3
    from .deepseek_v41 import install_deepseek_v41_compat
    from .k78_compat import install_k78_config_compat
    from .mixed_k_guard import install_mixed_k_prescan_guard
    from .runtime_policy import install_native_row_policy
    from .tp_geometry_compat import install_tp_geometry_compat
    from .uva_offload import install_uva_expert_validation

    # Accept K7/K8 config values before model-family/runtime wrappers inspect the
    # quantization config. ExLlamaV3's normal EXL3 MoE family covers K1-K8;
    # vllm-exl3's custom native p2b path remains independently gated to K2-K4.
    install_k78_config_compat(exl3)

    # Current vLLM keeps the authoritative expert TP/EP geometry under
    # RoutedExperts.moe_config.moe_parallel_config. Resolve that before falling
    # back to process-wide TP state so EP ranks keep whole expert matrices.
    install_tp_geometry_compat(exl3)

    # The low-memory mixed-K arena prescan assumes a contiguous linear expert
    # range. Disable that optimization for non-linear placement/EPLB and let the
    # authoritative vLLM loader mapping drive expert ownership instead.
    install_mixed_k_prescan_guard(exl3)

    # Install model-family compatibility before runtime policy wrappers inspect
    # the quantization config. vLLM still owns the DeepSeek V4.1 architecture.
    install_deepseek_v41_compat(exl3)
    install_native_row_policy(exl3)
    install_uva_expert_validation(exl3)


def runtime_diagnostics():
    """Return effective EXL3 runtime policy and extension availability."""
    from . import exl3
    from .deepseek_v41 import plan_deepseek_v41
    from .k78_compat import native_qualified_bits, supported_config_bits
    from .prefill_policy import grouped_prefill_enabled, grouped_prefill_max_rows
    from .runtime_policy import diagnostics, fused_temp_rows_requested, native_row_cap
    from .uva_offload import (
        EXL3_MOE_UVA_PARAMETER_SEGMENTS,
        uva_expert_offload_required,
    )

    native = exl3._load_native_exl3_ext()
    installed = bool(getattr(exl3, "_vllm_exl3_per_bit_policy_installed", False))
    caps = {
        bits: (
            exl3._native_moe_max_rows(bits)
            if installed
            else native_row_cap(bits, exl3._native_moe_max_rows)
        )
        for bits in native_qualified_bits()
    }
    record = diagnostics(
        backend=exl3.get_moe_kernel_backend(),
        native_available=native is not None,
        native_abi=(
            int(getattr(native, "P2B_MOE_ABI_VERSION", 0))
            if native is not None
            else 0
        ),
        native_caps=caps,
        fused_rows_actual=exl3.TEMP_ROWS_FUSED,
        fused_rows_requested=fused_temp_rows_requested(exl3.TEMP_ROWS_FUSED),
        fat_threshold=exl3.FAT_EXPERT_THRESHOLD,
        fat_kernel_available=exl3._fat_kernel_available(),
        spec_schedule=exl3.os.environ.get(
            exl3.SPECULATIVE_SCHEDULE_ENV, "<default>"
        ),
        per_bit_policy_installed=installed,
    )
    record["mixed_k"] = {
        "config_bits": list(supported_config_bits()),
        "exllamav3_moe_kernel_bits": list(range(1, 9)),
        "native_qualified_bits": list(native_qualified_bits()),
        "k78_config_compat_installed": bool(
            getattr(exl3, "_vllm_exl3_k78_compat_installed", False)
        ),
        "routed_allocation_scope": (
            "per-expert exact trellis shapes; uniform-K layers keep fused path"
        ),
        "tensor_level_mixed_k_within_layer": True,
        "heterogeneous_dispatch": "python_loop",
        "uniform_k_dispatch": "fused_when_available",
        "cudagraph_qualified": False,
        "recommended_first_boot": "eager",
        "arena_prescan_placement_contract": "linear_contiguous_global_expert_ids",
        "arena_prescan_guard_installed": bool(
            getattr(exl3, "_vllm_exl3_mixed_k_prescan_guard_installed", False)
        ),
        "legacy_shape_overlap_diagnostic": False,
        "legacy_shape_overlap_note": (
            "The old opt-in partial-copy trellis diagnostic is superseded by exact-shape "
            "per-expert trellis allocation. Non-trellis scale/marker shape mismatches "
            "remain hard failures."
        ),
        "note": (
            "K7/K8 are accepted for ExLlamaV3 execution. The custom native p2b "
            "path remains K2-K4. Routed experts store exact per-expert trellis "
            "shapes (including intra-expert w1/w2/w3 K disagreement). "
            "Heterogeneous packed K within a layer forces the LinearEXL3 "
            "python_loop; uniform-K layers still use the fused fast path. "
            "The heterogeneous path is correctness-first and should remain eager "
            "until CUDA-graph qualification is completed."
        ),
    }
    record["tp_geometry"] = {
        "nested_moe_config_compat_installed": bool(
            getattr(exl3, "_vllm_exl3_tp_geometry_compat_installed", False)
        ),
        "resolution_order": (
            "RoutedExperts.moe_config.moe_parallel_config -> legacy layer attrs -> process TP"
        ),
        "note": (
            "Expert-parallel layouts can have MoE tp_size=1 while process TP is >1; "
            "weight slicing must use the per-MoE geometry."
        ),
    }
    record["grouped_prefill"] = {
        "requested": grouped_prefill_enabled(),
        "supported_bits": [2, 3],
        "max_rows_per_window": grouped_prefill_max_rows(),
        "execution_available": False,
    }
    record["deepseek_v41"] = {
        "compat_installed": bool(
            getattr(exl3, "_vllm_exl3_v41_compat_installed", False)
        ),
        "recommended_tp4_ep4": plan_deepseek_v41().to_dict(),
    }
    record["cpu_offload"] = {
        "execution_available": False,
        "owner": "external ExLlamaV3 experiment only",
        "note": (
            "vllm-exl3 does not provide a host-CPU expert compute backend; "
            "use plan_exllamav3_cpu_offload() for that external runtime contract"
        ),
    }
    record["uva_expert_offload"] = {
        "requested": uva_expert_offload_required(),
        "guard_installed": bool(
            getattr(
                getattr(exl3, "Exl3MoEMethod", object),
                "process_weights_after_loading",
                None,
            )
            and bool(
                getattr(
                    getattr(
                        exl3.Exl3MoEMethod,
                        "process_weights_after_loading",
                        None,
                    ),
                    "_vllm_exl3_uva_guard_wrapped",
                    False,
                )
            )
        ),
        "parameter_segments": list(EXL3_MOE_UVA_PARAMETER_SEGMENTS),
        "execution_model": "GPU kernels over vLLM mapped pinned-host UVA views",
        "qualification": "experimental; requires real GPU parity/performance testing",
    }
    return record


def __getattr__(name: str):
    """Expose helpers without making the plugin entry point eager."""
    if name in {
        "get_speculative_draft_tokens",
        "parse_speculative_schedule",
        "is_adaptive_verification_enabled",
        "filter_speculative_candidates",
        "compute_mla_kv_cache_bytes",
        "validate_context_scaling",
    }:
        from . import exl3

        return getattr(exl3, name)
    if name in {
        "GroupedPrefillPlan",
        "grouped_prefill_enabled",
        "grouped_prefill_max_rows",
        "grouped_prefill_scratch_bytes",
        "plan_grouped_prefill",
    }:
        from . import prefill_policy

        return getattr(prefill_policy, name)
    if name in {
        "DeepseekV41Plan",
        "plan_deepseek_v41",
        "source_weight_block_size",
        "is_deepseek_v41_source_quant",
        "should_delegate_dspark_source",
    }:
        from . import deepseek_v41

        return getattr(deepseek_v41, name)
    if name in {"CpuOffloadPlan", "plan_exllamav3_cpu_offload"}:
        from . import cpu_offload

        return getattr(cpu_offload, name)
    if name in {
        "UvaExpertLayerStatus",
        "inspect_exl3_moe_uva_layer",
        "validate_exl3_moe_uva_layer",
        "uva_expert_offload_required",
    }:
        from . import uva_offload

        return getattr(uva_offload, name)
    if name in {"supported_config_bits", "native_qualified_bits"}:
        from . import k78_compat

        return getattr(k78_compat, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
