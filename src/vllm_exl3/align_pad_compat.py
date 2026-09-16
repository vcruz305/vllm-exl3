"""Ignore ``__align_pad__`` filler tensors when loading EXL3 packs.

Packs re-laid for zero-copy loading on unified-memory boxes (ExLlamaV3's
``EXL3_ATS_MMAP``, via ``util/align_safetensors.py``) carry small U8 gap fillers
named ``__align_pad__.<shard>.N`` so every real tensor starts on the alignment
grid the trellis kernels require. They are not weights and no module claims
them. ExLlamaV3's own loader skips the prefix; vLLM's ``AutoWeightsLoader``
raises ``ValueError`` on any key it cannot map to a parameter, so such a pack
fails partway through weight streaming.

``AutoWeightsLoader`` already supports this case through
``ignore_unexpected_prefixes``; the DeepSeek-V4.1 wrapper simply constructs the
loader without it. This installer wraps that model's ``load_weights`` so the
loader it builds ignores the pad prefix and nothing else: an unknown key that is
not a pad still raises, so a genuinely missing weight is never masked.
"""
from __future__ import annotations

from typing import Any

PAD_PREFIX = "__align_pad__"


def _patch_auto_weights_loader(models_utils: Any) -> bool:
    """Make AutoWeightsLoader ignore pad tensors, whoever constructs it."""
    loader_cls = getattr(models_utils, "AutoWeightsLoader", None)
    if loader_cls is None:
        return False
    if bool(getattr(loader_cls, "_vllm_exl3_align_pad_patched", False)):
        return True

    original_init = loader_cls.__init__

    def __init__(self, module, *, ignore_unexpected_prefixes=None, **kwargs):
        prefixes = list(ignore_unexpected_prefixes or [])
        if PAD_PREFIX not in prefixes:
            prefixes.append(PAD_PREFIX)
        original_init(self, module, ignore_unexpected_prefixes=prefixes, **kwargs)

    loader_cls.__init__ = __init__
    loader_cls._vllm_exl3_align_pad_patched = True
    return True


def install_align_pad_compat(exl3_module: Any) -> bool:
    """Install the pad-ignoring loader shim. Returns True when it is active."""
    if bool(getattr(exl3_module, "_vllm_exl3_align_pad_compat_installed", False)):
        return True
    try:
        from vllm.model_executor.models import utils as models_utils
    except Exception:
        return False
    ok = _patch_auto_weights_loader(models_utils)
    exl3_module._vllm_exl3_align_pad_compat_installed = bool(ok)
    return bool(ok)
