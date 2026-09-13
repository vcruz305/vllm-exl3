from __future__ import annotations

from types import SimpleNamespace

import vllm_exl3
import vllm_exl3.exl3 as exl3
from vllm_exl3.mixed_k_guard import install_mixed_k_prescan_guard


def test_mixed_k_runtime_contract_is_explicit() -> None:
    mixed = vllm_exl3.runtime_diagnostics()["mixed_k"]
    assert mixed["tensor_level_mixed_k_within_layer"] is True
    assert mixed["heterogeneous_dispatch"] == "python_loop"
    assert mixed["uniform_k_dispatch"] == "fused_when_available"
    assert mixed["cudagraph_qualified"] is False
    assert mixed["recommended_first_boot"] == "eager"
    assert mixed["arena_prescan_placement_contract"] == (
        "linear_contiguous_global_expert_ids"
    )
    assert mixed["arena_prescan_guard_installed"] is True
    assert mixed["legacy_shape_overlap_diagnostic"] is False


def test_prescan_guard_skips_non_linear_placement(monkeypatch) -> None:
    calls: list[int] = []

    def fake_prescan(layer, num_experts):
        calls.append(num_experts)
        return {"gate": {}, "up": {}, "down": {}}

    monkeypatch.setattr(exl3, "_try_prescan_trellis_shapes", fake_prescan)
    monkeypatch.delattr(exl3, "_vllm_exl3_mixed_k_prescan_guard_installed", raising=False)
    install_mixed_k_prescan_guard(exl3)

    layer = SimpleNamespace(expert_placement_strategy="round_robin")
    assert exl3._try_prescan_trellis_shapes(layer, 96) is None
    assert calls == []


def test_prescan_guard_allows_linear_placement(monkeypatch) -> None:
    expected = {"gate": {0: (1, 1, 48)}, "up": {}, "down": {}}

    def fake_prescan(layer, num_experts):
        assert num_experts == 96
        return expected

    monkeypatch.setattr(exl3, "_try_prescan_trellis_shapes", fake_prescan)
    monkeypatch.delattr(exl3, "_vllm_exl3_mixed_k_prescan_guard_installed", raising=False)
    install_mixed_k_prescan_guard(exl3)

    layer = SimpleNamespace(expert_placement_strategy="linear")
    assert exl3._try_prescan_trellis_shapes(layer, 96) == expected
