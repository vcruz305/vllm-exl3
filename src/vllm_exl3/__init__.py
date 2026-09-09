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
]


def register() -> None:
    # Importing the module executes its register_quantization_config decorator.
    from . import exl3 as _exl3  # noqa: F401


def runtime_diagnostics():
    """Return effective EXL3 runtime policy and extension availability."""
    from . import exl3
    from .runtime_policy import diagnostics, fused_temp_rows, native_row_cap

    native = exl3._load_native_exl3_ext()
    caps = {
        bits: native_row_cap(bits, exl3._native_moe_max_rows)
        for bits in (2, 3, 4)
    }
    return diagnostics(
        backend=exl3.get_moe_kernel_backend(),
        native_available=native is not None,
        native_abi=int(getattr(native, "P2B_MOE_ABI_VERSION", 0)) if native is not None else 0,
        native_caps=caps,
        fused_rows=fused_temp_rows(exl3.TEMP_ROWS_FUSED),
        fat_threshold=exl3.FAT_EXPERT_THRESHOLD,
        fat_kernel_available=exl3._fat_kernel_available(),
        spec_schedule=exl3.os.environ.get(exl3.SPECULATIVE_SCHEDULE_ENV, "<default>"),
    )


def __getattr__(name: str):
    """Expose scheduler helpers without making package import eager."""
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
