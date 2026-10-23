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

``VLLM_EXL3_DRAFT_VOCAB_META=1``
    When the DSpark draft checkpoint carries no ``embed_tokens``/``lm_head``
    tensors (vLLM shares the target's modules after the draft load), build
    the draft's own placeholders on the meta device instead of allocating
    two unused 200064 x 6144 BF16 tables (4.58 GiB) that are discarded a
    moment later. Draft checkpoints that do ship either table are loaded
    unchanged.

``VLLM_EXL3_TRIM_BEFORE_KV=1``
    After the startup profile run (which compiles the model for
    max_num_batched_tokens and leaves its activations cached by the torch
    allocator and its compile scratch in glibc arenas), trim before the
    worker allocates the KV cache and again before kernel warmup / CUDA graph
    capture. Installed lazily on the vLLM GPU Worker from the target
    post-load hook.

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


def trim_before_kv_enabled() -> bool:
    return _flag("VLLM_EXL3_TRIM_BEFORE_KV")


def _staged(fn: Any, tag: str) -> Any:
    @wraps(fn)
    def run(self, *args, **kwargs):
        if trim_before_kv_enabled():
            if probe_enabled():
                logger.info("%s", mem_line(f"PRE_{tag}_UNTRIMMED"))
            trim()
            STATS[f"trim_{tag.lower()}"] = STATS.get(f"trim_{tag.lower()}", 0) + 1
        if probe_enabled():
            logger.info("%s", mem_line(f"PRE_{tag}"))
        out = fn(self, *args, **kwargs)
        if probe_enabled():
            logger.info("%s", mem_line(f"POST_{tag}"))
        return out

    run._uma_staged = True
    return run


def install_worker_hooks(worker_cls: Any) -> bool:
    """Trim/probe around KV allocation and warmup (idempotent)."""
    done = False
    for attr, tag in (("initialize_from_config", "KV"), ("compile_or_warm_up_model", "WARMUP")):
        fn = getattr(worker_cls, attr, None)
        if fn is None or getattr(fn, "_uma_staged", False):
            continue
        setattr(worker_cls, attr, _staged(fn, tag))
        done = True
    return done


def _install_worker_hooks_from_loaded_modules() -> None:
    import sys

    mod = sys.modules.get("vllm.v1.worker.gpu_worker")
    cls = getattr(mod, "Worker", None) if mod is not None else None
    if cls is not None and install_worker_hooks(cls):
        logger.info("EXL3 UMA worker hooks installed on %s.%s", cls.__module__, cls.__name__)


# Draft architectures whose embed_tokens / lm_head are shared from the target
# when the draft checkpoint omits them (vLLM v1 dspark/utils.py
# load_dspark_model -> maybe_share_target_embed / lm_head share).
VOCAB_SHARED_DRAFT_ARCHS = frozenset({"DSparkMiniMaxDraftModel"})
_VOCAB_TABLE_MARKERS = ("embed_tokens", "lm_head")


def draft_vocab_meta_enabled() -> bool:
    return _flag("VLLM_EXL3_DRAFT_VOCAB_META")


def _safetensors_keys(path: str) -> list[str]:
    import json

    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        if n <= 0 or n > 256 * 2**20:
            raise ValueError(f"implausible safetensors header length {n}: {path}")
        header = json.loads(fh.read(n))
    return [k for k in header if k != "__metadata__"]


def checkpoint_has_vocab_tables(model_dir: str) -> bool | None:
    """True/False from the checkpoint's own tensor names; None if unreadable."""
    import glob

    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        return None
    try:
        for f in files:
            if any(m in k for k in _safetensors_keys(f) for m in _VOCAB_TABLE_MARKERS):
                return True
    except Exception as exc:
        logger.warning("draft vocab scan failed for %s: %r", model_dir, exc)
        return None
    return False


def _architectures(model_config: Any) -> list[str]:
    hf = getattr(model_config, "hf_config", None)
    return list(getattr(hf, "architectures", None) or [])


def is_vocab_shared_draft(model_config: Any) -> bool:
    if model_config is None:
        return False
    if not set(_architectures(model_config)) & VOCAB_SHARED_DRAFT_ARCHS:
        return False
    model_dir = str(getattr(model_config, "model", "") or "")
    if not model_dir or not os.path.isdir(model_dir):
        return False
    return checkpoint_has_vocab_tables(model_dir) is False


class _MetaUnquantizedEmbeddings:
    """Context: UnquantizedEmbeddingMethod.create_weights allocates on meta."""

    def __init__(self, method_cls: Any):
        self._cls = method_cls
        self._orig = None
        self.count = 0

    def __enter__(self):
        import torch

        orig = self._cls.create_weights
        self._orig = orig
        ctx = self

        def create_weights(method, layer, *args, **kwargs):
            with torch.device("meta"):
                out = orig(method, layer, *args, **kwargs)
            layer._exl3_meta_vocab_placeholder = True
            ctx.count += 1
            return out

        self._cls.create_weights = create_weights
        return self

    def __exit__(self, *exc):
        self._cls.create_weights = self._orig
        return False


def _model_config_arg(args: tuple, kwargs: dict) -> Any:
    if "model_config" in kwargs:
        return kwargs["model_config"]
    return args[1] if len(args) > 1 else None


def install_postload_hook(loader_cls: Any, embedding_method_cls: Any = None) -> None:
    orig_load = loader_cls.load_model
    if getattr(orig_load, "_uma_postload", False):
        return

    @wraps(orig_load)
    def load_model(self, *args, **kwargs):
        model_config = _model_config_arg(args, kwargs)
        draft = bool(set(_architectures(model_config)) & VOCAB_SHARED_DRAFT_ARCHS)
        meta_cls = embedding_method_cls
        if draft and draft_vocab_meta_enabled() and meta_cls is None:
            try:
                from vllm.model_executor.layers.vocab_parallel_embedding import (
                    UnquantizedEmbeddingMethod as meta_cls,
                )
            except Exception:
                meta_cls = None
        if draft and meta_cls is not None and draft_vocab_meta_enabled() and is_vocab_shared_draft(
            model_config
        ):
            with _MetaUnquantizedEmbeddings(meta_cls) as meta:
                model = orig_load(self, *args, **kwargs)
            STATS["draft_meta_vocab"] = STATS.get("draft_meta_vocab", 0) + meta.count
            logger.info(
                "EXL3 draft vocab tables on meta (shared from the target after load): %d",
                meta.count,
            )
        else:
            model = orig_load(self, *args, **kwargs)
        if postload_enabled():
            if draft:
                trim()  # the draft's tables are the target's; nothing to alias
                if probe_enabled():
                    logger.info("%s", mem_line("POST_DRAFT_LOAD"))
            else:
                postload(model)
        if not draft and (trim_before_kv_enabled() or probe_enabled()):
            _install_worker_hooks_from_loaded_modules()
        return model

    load_model._uma_postload = True
    loader_cls.load_model = load_model
