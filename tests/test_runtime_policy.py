import os

from vllm_exl3.runtime_policy import diagnostics, fused_temp_rows, native_row_cap


def test_fused_temp_rows_defaults_and_rejects_bad_values(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_FUSED_TEMP_ROWS", raising=False)
    assert fused_temp_rows(2048) == 2048
    monkeypatch.setenv("VLLM_EXL3_FUSED_TEMP_ROWS", "512")
    assert fused_temp_rows(2048) == 512
    monkeypatch.setenv("VLLM_EXL3_FUSED_TEMP_ROWS", "0")
    assert fused_temp_rows(2048) == 2048
    monkeypatch.setenv("VLLM_EXL3_FUSED_TEMP_ROWS", "bad")
    assert fused_temp_rows(2048) == 2048


def test_native_row_cap_is_per_bit(monkeypatch):
    for bits in (2, 3, 4):
        monkeypatch.delenv(f"VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K{bits}", raising=False)
    fallback = lambda bits: {2: 8, 3: 1, 4: 1}[bits]
    assert native_row_cap(2, fallback) == 8
    assert native_row_cap(3, fallback) == 1
    monkeypatch.setenv("VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K3", "4")
    assert native_row_cap(3, fallback) == 4
    assert native_row_cap(2, fallback) == 8


def test_diagnostics_schema_is_json_friendly():
    value = diagnostics(
        backend="native",
        native_available=True,
        native_abi=2,
        native_caps={4: 1, 2: 8, 3: 1},
        fused_rows=2048,
        fat_threshold=256,
        fat_kernel_available=True,
        spec_schedule="<default>",
    )
    assert value["native_row_caps"] == {"2": 8, "3": 1, "4": 1}
    assert value["native_abi"] == 2
    assert value["moe_backend"] == "native"
