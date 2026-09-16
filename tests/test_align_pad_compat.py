"""The pad shim must ignore ``__align_pad__`` keys and nothing else."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm_exl3.align_pad_compat import PAD_PREFIX, install_align_pad_compat

models_utils = pytest.importorskip("vllm.model_executor.models.utils")


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4))

    def load_weights(self, weights):
        loaded = set()
        params = dict(self.named_parameters())
        for name, tensor in weights:
            if name in params:
                params[name].data.copy_(tensor)
                loaded.add(name)
        return loaded


def _fresh_module():
    class _Shim:
        pass

    return _Shim()


def test_installer_reports_state() -> None:
    shim = _fresh_module()
    assert install_align_pad_compat(shim) is True
    assert shim._vllm_exl3_align_pad_compat_installed is True
    # Second install is a no-op that still reports active
    assert install_align_pad_compat(shim) is True


def test_pad_keys_are_ignored() -> None:
    install_align_pad_compat(_fresh_module())
    model = _Tiny()
    loader = models_utils.AutoWeightsLoader(model)
    loaded = loader.load_weights(
        [
            ("weight", torch.ones(4)),
            (f"{PAD_PREFIX}.model-00001-of-00017.0", torch.zeros(64, dtype=torch.uint8)),
        ]
    )
    assert "weight" in loaded
    assert torch.equal(model.weight.data, torch.ones(4))


def test_unknown_non_pad_key_still_raises() -> None:
    install_align_pad_compat(_fresh_module())
    model = _Tiny()
    loader = models_utils.AutoWeightsLoader(model)
    with pytest.raises(ValueError):
        loader.load_weights([("definitely_not_a_parameter", torch.ones(4))])


def test_explicit_prefixes_are_preserved() -> None:
    install_align_pad_compat(_fresh_module())
    model = _Tiny()
    loader = models_utils.AutoWeightsLoader(
        model, ignore_unexpected_prefixes=["vision."]
    )
    assert "vision." in loader.ignore_unexpected_prefixes
    assert PAD_PREFIX in loader.ignore_unexpected_prefixes
