"""Scoped checkpoint metadata input for constructor-only EXL3 storage planning.

The provider owns identity/bounds validation. This module does not open files,
materialize tensors, choose EP ownership, or set a quantization K.
"""
from contextlib import contextmanager
from contextvars import ContextVar

_provider = ContextVar("exl3_tensor_metadata_provider", default=None)

def current_tensor_metadata_provider():
    return _provider.get()

@contextmanager
def tensor_metadata_scope(provider):
    if not callable(provider):
        raise TypeError("tensor metadata provider must be callable")
    token = _provider.set(provider)
    try:
        yield
    finally:
        _provider.reset(token)
