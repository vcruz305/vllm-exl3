"""Load-time memory hygiene for unified-memory hosts (GB10 / DGX Spark).

On GB10 a CUDA allocation and a host anonymous page come out of the same
physical pool. Whatever the torch caching allocator or glibc keeps after a
tensor dies is therefore memory that the next MoE layer's expert arena cannot
use. The direct-fill loader already frees its own sources, but loader
temporaries elsewhere (scale/norm uploads, dtype conversions, freed
placeholders) stay cached until ``process_weights_after_loading`` runs at the
very end of the load -- after the load peak. With every routed expert of a
~109 GiB pack resident, that retention alone is the difference between a boot
and an OOM kill.

Switches (all default off; the recipe turns them on):

``VLLM_EXL3_TRIM_AFTER_LAYER=1``
    Before each MoE layer's arena allocation (``prepare_trellis_arena_plan``),
    run ``gc.collect()``, glibc ``malloc_trim(0)`` and
    ``torch.cuda.empty_cache()`` so the new arena is carved out of memory the
    previous layer's temporaries released.

``VLLM_EXL3_UMA_POSTLOAD=1``
    After ``DefaultModelLoader.load_model`` returns: alias the text embedding
    onto the pack shard through ATS (``ats_alias.alias_embedding``; needs
    ``VLLM_EXL3_ATS_ALIAS=1`` and the embedding at a 16-byte-aligned offset),
    then trim, so the CUDA copy of the embedding is gone before vLLM sizes
    the KV cache and loads the draft model. No routed expert is aliased
    unless an alias manifest asks for it.

``VLLM_EXL3_MEM_PROBE=1``
    One ``MEMPROBE`` log line per trim point (defaults on whenever one of the
    two switches above is on).
"""

from __future__ import annotations

import gc
import os
from functools import wraps
from typing import Any

try:  # vLLM's logger when available; stdlib otherwise (CPU-only CI)
    from vllm.logger import init_logger

    logger = init_logger("vllm." + __name__)
except Exception:  # pragma: no cover - exercised only without vLLM
    import logging

    logger = logging.getLogger(__name__)

_FALSY = ("", "0", "false", "False", "no")
STATS: dict[str, int] = {"layer_trims": 0, "postload_runs": 0, "embed_aliased": 0}


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in _FALSY


def trim_enabled() -> bool:
    return _flag("VLLM_EXL3_TRIM_AFTER_LAYER")


def postload_enabled() -> bool:
    return _flag("VLLM_EXL3_UMA_POSTLOAD")


def probe_enabled() -> bool:
    default = "1" if (trim_enabled() or postload_enabled()) else "0"
    return _flag("VLLM_EXL3_MEM_PROBE", default)


def _gib(kib: int) -> float:
    return kib / 1048576.0


def mem_line(tag: str) -> str:
    """One-line memory snapshot: torch CUDA allocator, /proc/meminfo, own RSS."""
    fields: dict[str, float] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                name, _, rest = line.partition(":")
                if name in ("MemAvailable", "MemFree", "AnonPages", "Cached", "Mapped"):
                    fields[name] = _gib(int(rest.split()[0]))
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(("VmRSS:", "RssAnon:", "RssFile:")):
                    key, _, rest = line.partition(":")
                    fields[key] = _gib(int(rest.split()[0]))
    except OSError:
        pass
    alloc = reserved = -1.0
    try:
        import torch

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 2**30
            reserved = torch.cuda.memory_reserved() / 2**30
    except Exception:
        pass
    return (
        f"MEMPROBE {tag} cuda_alloc={alloc:.2f}GiB cuda_reserved={reserved:.2f}GiB "
        f"avail={fields.get('MemAvailable', -1):.2f}GiB free={fields.get('MemFree', -1):.2f}GiB "
        f"anon={fields.get('AnonPages', -1):.2f}GiB cached={fields.get('Cached', -1):.2f}GiB "
        f"mapped={fields.get('Mapped', -1):.2f}GiB rss={fields.get('VmRSS', -1):.2f}GiB "
        f"rss_anon={fields.get('RssAnon', -1):.2f}GiB rss_file={fields.get('RssFile', -1):.2f}GiB"
    )


def tensor_census(top: int = 6, device: str = "cpu") -> str:
    """Live CPU tensors by unique storage (VLLM_EXL3_MEM_PROBE_TENSORS=1 only).

    Counts every CPU torch tensor the garbage collector can see, once per
    storage, split into pinned and pageable, with the largest (shape, dtype)
    groups. Storages that view a mapped file are counted too; the group shapes
    tell them apart from anonymous copies.
    """
    import collections

    try:
        import torch
    except Exception:
        return "TENSORS unavailable"
    seen: set[int] = set()
    total = pinned = 0
    by_key: collections.Counter = collections.Counter()
    n_key: collections.Counter = collections.Counter()
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, torch.Tensor) or obj.device.type != device:
                continue
            st = obj.untyped_storage()
            ptr = int(st.data_ptr())
            if ptr == 0 or ptr in seen:
                continue
            seen.add(ptr)
            nbytes = int(st.nbytes())
            total += nbytes
            if device == "cpu" and obj.is_pinned():
                pinned += nbytes
            key = (tuple(int(x) for x in obj.shape[:4]), str(obj.dtype).replace("torch.", ""))
            by_key[key] += nbytes
            n_key[key] += 1
        except Exception:
            continue
    tops = ", ".join(
        f"{k[0]}:{k[1]}x{n_key[k]}={v / 2**20:.0f}MiB" for k, v in by_key.most_common(top)
    )
    host = ""
    try:
        hs = torch.cuda.host_memory_stats()
        host = (
            f" host_alloc={hs.get('allocated_bytes.current', hs.get('allocated_bytes', 0)) / 2**30:.2f}GiB"
            f" host_reserved={hs.get('reserved_bytes.current', hs.get('reserved_bytes', 0)) / 2**30:.2f}GiB"
        )
    except Exception:
        pass
    p1 = df = ""
    try:
        from . import exl3 as _ex

        p1 = f" p1={dict(_ex._P1_STATS)}"
        df = f" fill_calls={_ex._DIRECT_FILL_STATS.get('DIRECT_FILL_CALLS')}"
    except Exception:
        pass
    return (
        f"TENSORS {device}_storages={len(seen)} {device}={total / 2**30:.2f}GiB "
        f"pinned={pinned / 2**30:.2f}GiB{host}{p1}{df} top=[{tops}]"
    )


def trim() -> None:
    """Return freed host and CUDA blocks to the shared pool."""
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _layer_name(layer: Any) -> str:
    return str(getattr(layer, "layer_name", None) or getattr(layer, "prefix", None) or "?")


def postload(model: Any) -> int:
    """Embedding alias + trim after the model load; returns aliased module count."""
    STATS["postload_runs"] += 1
    if probe_enabled():
        logger.info("%s", mem_line("POST_LOAD"))
        if _flag("VLLM_EXL3_MEM_PROBE_TENSORS"):
            logger.info("%s POST_LOAD", tensor_census())
    n = 0
    try:
        from . import ats_alias

        n = int(ats_alias.alias_embedding(model) or 0)
    except Exception as exc:  # the optional path never fails a boot
        logger.warning("UMA post-load embedding alias failed: %r", exc)
    STATS["embed_aliased"] += n
    trim()
    if probe_enabled():
        logger.info("%s embed_modules=%d", mem_line("POST_ALIAS"), n)
    return n


def install_uma_load_memory(exl3: Any) -> None:
    """Wrap the arena allocator and the model loader (idempotent)."""
    orig_plan = getattr(exl3, "prepare_trellis_arena_plan", None)
    if orig_plan is not None and not getattr(orig_plan, "_uma_trim", False):

        @wraps(orig_plan)
        def prepare_trellis_arena_plan(layer, shapes_by_proj, *args, **kwargs):
            if trim_enabled():
                trim()
                STATS["layer_trims"] += 1
                if probe_enabled():
                    logger.info("%s layer=%s", mem_line("PRE_ARENA"), _layer_name(layer))
                    if _flag("VLLM_EXL3_MEM_PROBE_TENSORS") and STATS["layer_trims"] % 8 == 1:
                        logger.info("%s layer=%s", tensor_census(), _layer_name(layer))
                        if STATS["layer_trims"] == 1:
                            logger.info(
                                "%s layer=%s",
                                tensor_census(top=24, device="cuda"),
                                _layer_name(layer),
                            )
            stop_at = int(os.environ.get("VLLM_EXL3_MEM_PROBE_ABORT_AFTER", "0") or 0)
            if stop_at and STATS["layer_trims"] >= stop_at:
                raise RuntimeError(
                    f"VLLM_EXL3_MEM_PROBE_ABORT_AFTER={stop_at}: diagnostic load stop"
                )
            return orig_plan(layer, shapes_by_proj, *args, **kwargs)

        prepare_trellis_arena_plan._uma_trim = True
        exl3.prepare_trellis_arena_plan = prepare_trellis_arena_plan

    try:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    except Exception:  # no vLLM (CPU-only CI): nothing to wrap
        return
    install_postload_hook(DefaultModelLoader)


def install_postload_hook(loader_cls: Any) -> None:
    orig_load = loader_cls.load_model
    if getattr(orig_load, "_uma_postload", False):
        return

    @wraps(orig_load)
    def load_model(self, *args, **kwargs):
        model = orig_load(self, *args, **kwargs)
        if postload_enabled():
            postload(model)
        return model

    load_model._uma_postload = True
    loader_cls.load_model = load_model
