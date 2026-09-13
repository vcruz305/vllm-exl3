from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_exl3.physical_k_compat import install_physical_fused_k_compat


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def info_once(self, *args, **kwargs):
        pass


def _pack(gate: int, up: int, down: int):
    return {
        "gate": SimpleNamespace(K=gate),
        "up": SimpleNamespace(K=up),
        "down": SimpleNamespace(K=down),
    }


def test_uniform_physical_k_overrides_config_k() -> None:
    def original(layer, inners):
        # Reproduce the pre-fix behavior of the core helper.
        layer._exl3_k = int(layer._exl3_bits)

    module = SimpleNamespace(build_exl3_fused_state=original, logger=_Logger())
    install_physical_fused_k_compat(module)

    layer = SimpleNamespace(_exl3_bits=4)
    module.build_exl3_fused_state(layer, [_pack(7, 7, 7), _pack(7, 7, 7)])

    assert layer._exl3_k == 7
    assert layer._exl3_physical_fused_k == 7
    assert layer._exl3_configured_k == 4


def test_uniform_physical_k_matching_config_stays_same() -> None:
    def original(layer, inners):
        layer._exl3_k = int(layer._exl3_bits)

    module = SimpleNamespace(build_exl3_fused_state=original, logger=_Logger())
    install_physical_fused_k_compat(module)

    layer = SimpleNamespace(_exl3_bits=4)
    module.build_exl3_fused_state(layer, [_pack(4, 4, 4)])

    assert layer._exl3_k == 4
    assert layer._exl3_physical_fused_k == 4


def test_heterogeneous_physical_k_cannot_build_fused_state() -> None:
    calls = []

    def original(layer, inners):
        calls.append(True)

    module = SimpleNamespace(build_exl3_fused_state=original, logger=_Logger())
    install_physical_fused_k_compat(module)

    with pytest.raises(RuntimeError, match="requires one physical K"):
        module.build_exl3_fused_state(
            SimpleNamespace(_exl3_bits=4), [_pack(3, 4, 3)]
        )
    assert calls == []


def test_runtime_register_installs_physical_k_guard() -> None:
    import vllm_exl3

    vllm_exl3.register()
    mixed = vllm_exl3.runtime_diagnostics()["mixed_k"]
    assert mixed["physical_fused_k_guard_installed"] is True
    assert mixed["fused_k_source"] == "physical_trellis_geometry"
