from types import SimpleNamespace

from vllm_exl3.runtime_policy import (
    diagnostics,
    fused_temp_rows_requested,
    install_native_row_policy,
    native_row_cap,
)


def test_fused_temp_rows_is_advisory_only(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_FUSED_TEMP_ROWS", raising=False)
    assert fused_temp_rows_requested(2048) == 2048
    monkeypatch.setenv("VLLM_EXL3_FUSED_TEMP_ROWS", "512")
    assert fused_temp_rows_requested(2048) == 512
    monkeypatch.setenv("VLLM_EXL3_FUSED_TEMP_ROWS", "0")
    assert fused_temp_rows_requested(2048) == 2048
    monkeypatch.setenv("VLLM_EXL3_FUSED_TEMP_ROWS", "bad")
    assert fused_temp_rows_requested(2048) == 2048


def test_native_row_cap_is_per_bit(monkeypatch):
    for bits in (2, 3, 4):
        monkeypatch.delenv(f"VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K{bits}", raising=False)
    fallback = lambda bits: {2: 8, 3: 1, 4: 1}[bits]
    assert native_row_cap(2, fallback) == 8
    assert native_row_cap(3, fallback) == 1
    monkeypatch.setenv("VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K3", "4")
    assert native_row_cap(3, fallback) == 4
    assert native_row_cap(2, fallback) == 8
    assert native_row_cap(4, fallback) == 1


def test_install_native_row_policy_wraps_actual_resolver_once(monkeypatch):
    for bits in (2, 3, 4):
        monkeypatch.delenv(f"VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K{bits}", raising=False)
    module = SimpleNamespace(_native_moe_max_rows=lambda bits: {2: 8, 3: 1, 4: 1}[bits])
    assert install_native_row_policy(module) is True
    assert install_native_row_policy(module) is False
    assert module._native_moe_max_rows(2) == 8
    assert module._native_moe_max_rows(3) == 1

    monkeypatch.setenv("VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K3", "4")
    assert module._native_moe_max_rows(3) == 4
    assert module._native_moe_max_rows(2) == 8
    assert module._vllm_exl3_per_bit_policy_installed is True
    assert callable(module._vllm_exl3_native_row_cap_base)


def test_diagnostics_schema_separates_actual_and_requested_scratch():
    value = diagnostics(
        backend="native",
        native_available=True,
        native_abi=2,
        native_caps={4: 1, 2: 8, 3: 1},
        fused_rows_actual=2048,
        fused_rows_requested=512,
        fat_threshold=256,
        fat_kernel_available=True,
        spec_schedule="<default>",
        per_bit_policy_installed=True,
    )
    assert value["native_row_caps"] == {"2": 8, "3": 1, "4": 1}
    assert value["native_abi"] == 2
    assert value["moe_backend"] == "native"
    assert value["per_bit_native_policy_installed"] is True
    assert value["fused_temp_rows_actual"] == 2048
    assert value["fused_temp_rows_requested"] == 512
    assert value["fused_temp_rows_override_active"] is False
