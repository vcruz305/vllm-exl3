"""Tests for dense EXL3 call routing (GEMV, reconstruct, or sliced paths)."""

import pytest

pytest.importorskip("torch")
import torch
import vllm_exl3.exl3 as exl3


class FakeLinear:
    """Fake linear object that records forward calls and returns deterministic output."""

    def __init__(self):
        self.calls = []

    def forward(self, x, params, out_dtype):
        """Record call and return deterministic output."""
        rows = x.shape[0]
        self.calls.append((rows, dict(params)))
        # Return shape [rows, 4] for any input, deterministic
        return torch.full((rows, 4), float(rows), dtype=out_dtype)


def test_rows_1_and_2_use_gemv():
    """Rows 1 and 2 should use GEMV (single call with no params)."""
    for rows in [1, 2]:
        linear = FakeLinear()
        x = torch.randn(rows, 4, dtype=torch.float16)
        result = exl3._dense_forward(linear, x)

        assert len(linear.calls) == 1
        assert linear.calls[0] == (rows, {})
        assert result.shape == (rows, 4)


def test_rows_3_slices_through_gemv():
    """Rows 3 should slice through 2-row GEMV chunks."""
    linear = FakeLinear()
    x = torch.randn(3, 4, dtype=torch.float16)
    result = exl3._dense_forward(linear, x)

    assert len(linear.calls) == 2
    assert linear.calls[0] == (2, {})
    assert linear.calls[1] == (1, {})
    assert result.shape == (3, 4)
    # Check concatenation: first chunk rows are 2, second chunk row is 1
    expected = torch.cat([torch.full((2, 4), 2.0), torch.full((1, 4), 1.0)])
    assert torch.allclose(result, expected)


def test_rows_8_slices_with_gemv_max_2():
    """Rows 8 should slice into 4 calls of 2 rows each (with monkeypatch recon_min=9)."""
    # Save original and set high threshold so rows 8 doesn't reconstruct
    original_recon = exl3._EXL3_RECON_MIN_ROWS
    exl3._EXL3_RECON_MIN_ROWS = 9

    try:
        linear = FakeLinear()
        x = torch.randn(8, 4, dtype=torch.float16)
        result = exl3._dense_forward(linear, x)

        assert len(linear.calls) == 4
        for i, (rows, params) in enumerate(linear.calls):
            assert rows == 2
            assert params == {}
        assert result.shape == (8, 4)
    finally:
        exl3._EXL3_RECON_MIN_ROWS = original_recon


def test_rows_9_and_144_use_reconstruct():
    """Rows 9 and 144 should use reconstruct+hgemm."""
    for rows in [9, 144]:
        linear = FakeLinear()
        x = torch.randn(rows, 4, dtype=torch.float16)
        result = exl3._dense_forward(linear, x)

        assert len(linear.calls) == 1
        assert linear.calls[0] == (rows, {"reconstruct": True})
        assert result.shape == (rows, 4)


def test_rows_145_bypass_reconstruct():
    """Rows 145+ bypass the reconstruct logic (exllamav3 handles it internally)."""
    linear = FakeLinear()
    x = torch.randn(145, 4, dtype=torch.float16)
    result = exl3._dense_forward(linear, x)

    assert len(linear.calls) == 1
    assert linear.calls[0] == (145, {})
    assert result.shape == (145, 4)


def test_coop_gemm_flag_restores_old_dispatch():
    """VLLM_EXL3_COOP_GEMM=True should allow cooperative GEMM for all rows."""
    original_coop = exl3._EXL3_COOP_GEMM
    exl3._EXL3_COOP_GEMM = True

    try:
        linear = FakeLinear()
        x = torch.randn(8, 4, dtype=torch.float16)
        result = exl3._dense_forward(linear, x)

        # With coop flag on, should use single call with no params
        assert len(linear.calls) == 1
        assert linear.calls[0] == (8, {})
    finally:
        exl3._EXL3_COOP_GEMM = original_coop


def test_env_int_returns_9_when_unset(monkeypatch):
    """_env_int should return default 9 when env var is unset."""
    monkeypatch.delenv("VLLM_EXL3_RECONSTRUCT_MIN_ROWS", raising=False)
    result = exl3._env_int("VLLM_EXL3_RECONSTRUCT_MIN_ROWS", 9)
    assert result == 9


def test_env_int_returns_integer_when_set(monkeypatch):
    """_env_int should return the integer when env var is set to a valid int."""
    monkeypatch.setenv("VLLM_EXL3_RECONSTRUCT_MIN_ROWS", "15")
    result = exl3._env_int("VLLM_EXL3_RECONSTRUCT_MIN_ROWS", 9)
    assert result == 15


def test_env_int_returns_default_when_garbage(monkeypatch):
    """_env_int should return default when env var is set to garbage."""
    monkeypatch.setenv("VLLM_EXL3_RECONSTRUCT_MIN_ROWS", "not_an_int")
    result = exl3._env_int("VLLM_EXL3_RECONSTRUCT_MIN_ROWS", 9)
    assert result == 9
