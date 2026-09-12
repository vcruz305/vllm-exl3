from __future__ import annotations

import pytest

import vllm_exl3
from vllm_exl3 import exl3
from vllm_exl3.k78_compat import native_qualified_bits, supported_config_bits


def test_k78_config_acceptance_and_restore() -> None:
    vllm_exl3.register()
    cfg = exl3.Exl3Config(
        bits=8,
        layer_bits={"0": 7, "1": 2, "2": 8},
    )
    assert cfg.bits == 8
    assert cfg.layer_bits == {0: 7, 1: 2, 2: 8}
    assert cfg.bits_for_prefix("model.layers.0.ffn.experts") == 7
    assert cfg.bits_for_prefix("model.layers.2.ffn.experts") == 8
    assert cfg.bits_for_prefix("model.layers.9.ffn.experts") == 8


def test_k78_non_routed_config_is_restored() -> None:
    vllm_exl3.register()
    nr = {
        "bits": 8,
        "layer_bits": {"o_proj": 7},
        "layers": {"model.layers.0.shared_experts.down_proj": {"bits": 8}},
    }
    cfg = exl3.Exl3Config(bits=7, non_routed_exl3=nr)
    assert cfg.bits == 7
    assert cfg.non_routed_exl3 == nr


def test_config_still_rejects_outside_k2_k8() -> None:
    vllm_exl3.register()
    with pytest.raises(ValueError, match="K2-K8"):
        exl3.Exl3Config(bits=9)
    with pytest.raises(ValueError, match="K2-K8"):
        exl3.Exl3Config(bits=1)


def test_native_bits_are_not_widened_by_config_compat() -> None:
    assert supported_config_bits() == (2, 3, 4, 5, 6, 7, 8)
    assert native_qualified_bits() == (2, 3, 4)
