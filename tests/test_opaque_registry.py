"""The opaque-layer registry must be a real dict on every torch version."""

from __future__ import annotations

import types

import pytest

exl3 = pytest.importorskip("vllm_exl3.exl3")


class _AutoNamespaceHolder(types.ModuleType):
    """Mimics torch 2.14 ``torch.ops``: any missing attribute materializes."""

    def __getattr__(self, name):
        ns = types.SimpleNamespace(name=name)
        setattr(self, name, ns)
        return ns


def test_process_dict_ignores_auto_materialized_attributes():
    holder = _AutoNamespaceHolder("ops")
    assert hasattr(holder, "_exl3_opaque_layers")  # the trap the old guard fell into
    reg = exl3._process_dict(holder, "_exl3_opaque_layers")
    assert isinstance(reg, dict)
    reg["x"] = 1
    assert exl3._process_dict(holder, "_exl3_opaque_layers") is reg


def test_module_registries_are_dicts_and_accept_layers():
    torch = pytest.importorskip("torch")
    assert isinstance(exl3._EXL3_OPAQUE_LAYERS, dict)
    assert isinstance(exl3._EXL3_OPAQUE_METHODS, dict)
    layer = torch.nn.Module()
    layer._exl3_prefix = "model.layers.0.mlp.gate_up_proj"
    name = exl3._exl3_register_opaque_layer(layer, "linear", object())
    assert exl3._EXL3_OPAQUE_LAYERS[name] is layer
    assert vars(torch)["_exl3_opaque_layers"] is exl3._EXL3_OPAQUE_LAYERS
