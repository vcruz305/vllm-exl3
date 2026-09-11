from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_exl3.uva_offload import (
    EXL3_MOE_UVA_PARAMETER_SEGMENTS,
    inspect_exl3_moe_uva_layer,
    install_uva_expert_validation,
    validate_exl3_moe_uva_layer,
)


class _Param:
    def __init__(self, device_type: str = "cuda", uva: bool = False):
        self.device = SimpleNamespace(type=device_type)
        if uva:
            self._vllm_is_uva_offloaded = True


class _Layer:
    def __init__(
        self,
        *,
        uva_names: set[str] | None = None,
        cpu_names: set[str] | None = None,
    ):
        uva_names = uva_names or set()
        cpu_names = cpu_names or set()
        for name in EXL3_MOE_UVA_PARAMETER_SEGMENTS:
            setattr(
                self,
                name,
                _Param(
                    device_type="cpu" if name in cpu_names else "cuda",
                    uva=name in uva_names,
                ),
            )


def test_resident_layer_is_not_uva():
    status = inspect_exl3_moe_uva_layer(_Layer(), required=False)
    assert status.applicable
    assert not status.fully_uva_offloaded
    assert not status.partially_uva_offloaded
    assert set(status.resident_parameters) == set(EXL3_MOE_UVA_PARAMETER_SEGMENTS)


def test_complete_mapped_payload_is_accepted():
    names = set(EXL3_MOE_UVA_PARAMETER_SEGMENTS)
    status = validate_exl3_moe_uva_layer(_Layer(uva_names=names))
    assert status.fully_uva_offloaded
    assert not status.partially_uva_offloaded
    assert not status.cpu_fallback_parameters


def test_partial_uva_is_rejected():
    names = {EXL3_MOE_UVA_PARAMETER_SEGMENTS[0]}
    with pytest.raises(RuntimeError, match="partial"):
        validate_exl3_moe_uva_layer(_Layer(uva_names=names))


def test_non_uva_cpu_fallback_is_rejected():
    cpu = {EXL3_MOE_UVA_PARAMETER_SEGMENTS[0]}
    with pytest.raises(RuntimeError, match="ordinary CPU tensors"):
        validate_exl3_moe_uva_layer(_Layer(cpu_names=cpu))


def test_install_guard_runs_before_original_post_load(monkeypatch):
    names = set(EXL3_MOE_UVA_PARAMETER_SEGMENTS)
    layer = _Layer(uva_names=names)
    events: list[str] = []

    class FakeMethod:
        def process_weights_after_loading(self, target):
            events.append("original")
            assert hasattr(target, "_exl3_uva_expert_status")
            return "ok"

    fake_module = SimpleNamespace(Exl3MoEMethod=FakeMethod)
    monkeypatch.setenv("VLLM_EXL3_REQUIRE_UVA_EXPERTS", "1")
    install_uva_expert_validation(fake_module)

    result = FakeMethod().process_weights_after_loading(layer)
    assert result == "ok"
    assert events == ["original"]
    assert layer._exl3_uva_expert_status["fully_uva_offloaded"] is True


def test_guard_is_noop_when_not_requested(monkeypatch):
    layer = _Layer()
    events: list[str] = []

    class FakeMethod:
        def process_weights_after_loading(self, target):
            events.append("original")
            return "ok"

    fake_module = SimpleNamespace(Exl3MoEMethod=FakeMethod)
    monkeypatch.delenv("VLLM_EXL3_REQUIRE_UVA_EXPERTS", raising=False)
    install_uva_expert_validation(fake_module)

    assert FakeMethod().process_weights_after_loading(layer) == "ok"
    assert events == ["original"]
    assert not hasattr(layer, "_exl3_uva_expert_status")
