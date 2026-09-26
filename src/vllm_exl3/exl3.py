"""Portions of this module derive from Mia's AI Lab, overlay/exl3.py in
GLM-5.3-Flash-EXL3-2x-DGX-Sparks, first published 2026-08-27, which precedes this
project. The routed-expert EXL3/MCG path, its pointer-table construction, expert-map
pinning and diagnostic strings originate there.

Copyright (c) 2026 Mia's AI Lab. MIT. See THIRD_PARTY_NOTICES.md.

The EXL3 trellis format, the MCG codebook and the quantization method are ExLlamaV3's
work, Copyright (c) 2025 Turboderp, MIT. See THIRD_PARTY_NOTICES.md.
"""

# SPDX-License-Identifier: Apache-2.0
# EXL3 trellis quantization for routed experts, dense linears
# (non_routed_exl3), lm_head (ParallelLMHead) and row-wise n-gram embedding
# tables (ngram_embedding).
#
# Codebooks are mcg or mul1. Per-tensor K comes from layer_bits and
# non_routed_exl3.layers; matrices are padded to multiples of 128.
# Non-routed tensors without a spec stay native (UnquantizedLinearMethod).
#
# Experts never expand to a persistent BF16 weight; LinearEXL3 /
# exllamav3_ext runs the trellis GEMM. TP=2 shards gate/up column-wise and
# down row-wise; the MoE runner all-reduces the combined output.

from __future__ import annotations

import gc
import importlib
import json
import math
import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import re
try:
    import torch
    import torch.nn.functional as F
    from torch.nn.parameter import Parameter
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Parameter = None  # type: ignore[assignment]

try:
    from vllm.logger import init_logger
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )
    from vllm.model_executor.layers.linear import (
        LinearBase,
        LinearMethodBase,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
        QuantizeMethodBase,
    )
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.utils import set_weight_attrs
    _VLLM_AVAILABLE = True
    # Under the "vllm." hierarchy so vLLM's logging config actually emits these
    # INFO lines; a bare module name is dropped and the load log shows nothing.
    logger = init_logger("vllm." + __name__)
except ImportError:
    import logging
    _VLLM_AVAILABLE = False
    logger = logging.getLogger("vllm." + __name__)

    class FusedMoEQuantConfig:  # type: ignore[no-redef]
        pass

    class FusedMoEMethodBase:  # type: ignore[no-redef]
        pass

    class LinearBase:  # type: ignore[no-redef]
        pass

    class LinearMethodBase:  # type: ignore[no-redef]
        pass

    class UnquantizedLinearMethod:  # type: ignore[no-redef]
        pass

    class QuantizationConfig:  # type: ignore[no-redef]
        pass

    class QuantizeMethodBase:  # type: ignore[no-redef]
        pass

    def register_quantization_config(name: str):  # type: ignore[no-redef]
        def decorator(cls):
            return cls
        return decorator

    def set_weight_attrs(param, attrs):  # type: ignore[no-redef]
        for k, v in attrs.items():
            setattr(param, k, v)

MCG_MULTIPLIER = 0xCBAC1FED
MCG_MARKER_SIGNED_INT32 = -877912083
MUL1_MULTIPLIER = 0x83DCD12D
MUL1_MARKER_SIGNED_INT32 = -2082680531
EXL3_SUFFIXES = ("trellis", "suh", "svh", "mcg", "mul1")
SWIGLU_LIMIT_DEFAULT = 10.0
TEMP_ROWS_FUSED = 2048
_COOP = os.environ.get("VLLM_EXL3_COOP", "0") == "1"
try:
    FAT_EXPERT_THRESHOLD = max(0, int(os.environ.get("VLLM_EXL3_FAT_THRESHOLD", "256")))
except (TypeError, ValueError):
    FAT_EXPERT_THRESHOLD = 256
MOE_ACT_SILU = 0
# Shared fused scratch: decode is sequential across layers.
_FUSED_TEMP_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

# The schedule is deliberately expressed as inclusive ranges.  Keeping the
# policy here (rather than in a serving script) lets vLLM callers use the same
# batch-adaptive behaviour regardless of how the plugin is loaded.
DEFAULT_SPECULATIVE_SCHEDULE: list[tuple[int, int, int]] = [
    (1, 4, 3),
    (5, 8, 2),
    (9, 16, 1),
]
SPECULATIVE_SCHEDULE_ENV = "VLLM_EXL3_SPEC_SCHEDULE"
ADAPTIVE_VERIFICATION_ENV = "VLLM_EXL3_ADAPTIVE_VERIFICATION"


def compute_mla_kv_cache_bytes(
    context_len: int,
    num_layers: int = 43,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    dtype_bytes: int = 1,
) -> int:
    """Return the FP8 MLA KV-cache footprint for ``context_len`` tokens.

    DeepSeek-V4 stores one compressed KV latent and one decoupled RoPE key per
    layer.  The calculation is intentionally integer-only so callers can use
    it for an exact allocation or admission decision before starting a boot.
    """
    values = {
        "context_len": context_len,
        "num_layers": num_layers,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "dtype_bytes": dtype_bytes,
    }
    normalized: dict[str, int] = {}
    for name, value in values.items():
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must be an integer") from exc
        if integer != value:
            raise ValueError(f"{name} must be an integer")
        if integer < 0:
            raise ValueError(f"{name} must be non-negative")
        normalized[name] = integer

    return (
        normalized["context_len"]
        * normalized["num_layers"]
        * (normalized["kv_lora_rank"] + normalized["qk_rope_head_dim"])
        * normalized["dtype_bytes"]
    )


def _env_float_override(default: float, *names: str, minimum: float | None = None) -> float:
    """Read the first valid finite float from a list of environment names."""
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            value = float(raw.strip())
        except (AttributeError, TypeError, ValueError):
            continue
        if not math.isfinite(value) or (minimum is not None and value < minimum):
            continue
        return value
    return default


def _env_int_override(default: int, *names: str, minimum: int | None = None) -> int:
    """Read the first valid integer from a list of environment names."""
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            value = int(raw.strip(), 10)
        except (AttributeError, TypeError, ValueError):
            continue
        if minimum is not None and value < minimum:
            continue
        return value
    return default


def validate_context_scaling(
    max_model_len: int,
    model_weights_gb: float = 95.4,
    total_mem_gb: float = 128.0,
    mem_util: float = 0.90,
    chunk_size: int = 2048,
) -> dict[str, float | int | bool]:
    """Validate an MLA context ceiling against a unified-memory budget.

    ``VLLM_EXL3_CONTEXT_*`` variables are the canonical overrides.  Shorter
    ``VLLM_EXL3_*`` aliases are accepted for shell compatibility.  Invalid
    values are ignored and leave the corresponding function argument intact.
    ``chunk_size`` is reported so callers can associate the result with their
    chunked-prefill configuration; it does not alter the static KV footprint.
    """
    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int):
        raise ValueError("max_model_len must be a positive integer")
    if max_model_len <= 0:
        raise ValueError("max_model_len must be positive")

    model_weights_gb = _env_float_override(
        float(model_weights_gb),
        "VLLM_EXL3_CONTEXT_MODEL_WEIGHTS_GB",
        "VLLM_EXL3_MODEL_WEIGHTS_GB",
    )
    total_mem_gb = _env_float_override(
        float(total_mem_gb),
        "VLLM_EXL3_CONTEXT_TOTAL_MEM_GB",
        "VLLM_EXL3_TOTAL_MEM_GB",
    )
    mem_util = _env_float_override(
        float(mem_util),
        "VLLM_EXL3_CONTEXT_MEM_UTIL",
        "VLLM_EXL3_MEM_UTIL",
    )
    chunk_size = _env_int_override(
        int(chunk_size),
        "VLLM_EXL3_CONTEXT_CHUNK_SIZE",
        "VLLM_EXL3_CHUNK_SIZE",
        minimum=1,
    )

    if not math.isfinite(model_weights_gb) or model_weights_gb < 0:
        raise ValueError("model_weights_gb must be finite and non-negative")
    if not math.isfinite(total_mem_gb) or total_mem_gb <= 0:
        raise ValueError("total_mem_gb must be finite and positive")
    if not math.isfinite(mem_util) or not 0.0 < mem_util <= 1.0:
        raise ValueError("mem_util must be finite and in (0.0, 1.0]")

    kv_cache_bytes = compute_mla_kv_cache_bytes(max_model_len)
    kv_cache_gb = kv_cache_bytes / (1024**3)
    usable_mem_gb = total_mem_gb * mem_util
    available_headroom_gb = usable_mem_gb - model_weights_gb - kv_cache_gb
    safety_margin_gb = total_mem_gb - model_weights_gb - kv_cache_gb
    return {
        "max_model_len": max_model_len,
        "kv_cache_bytes": kv_cache_bytes,
        "kv_cache_gb": kv_cache_gb,
        "usable_mem_gb": usable_mem_gb,
        "available_headroom_gb": available_headroom_gb,
        "fits": available_headroom_gb > 0.0,
        "safety_margin_gb": safety_margin_gb,
        "chunk_size": chunk_size,
    }


def _validated_speculative_schedule(schedule: object) -> list[tuple[int, int, int]] | None:
    """Return a normalized schedule, or ``None`` when it is invalid.

    A schedule with overlapping ranges is ambiguous, so it is rejected rather
    than silently depending on item order.  Gaps are valid and intentionally
    return zero draft tokens for the uncovered batch sizes.
    """
    if not isinstance(schedule, (list, tuple)) or not schedule:
        return None

    normalized: list[tuple[int, int, int]] = []
    for entry in schedule:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            return None
        values: list[int] = []
        for value in entry:
            # Do not silently truncate floats (or accept booleans, which are
            # ``int`` subclasses) in a serving policy supplied by a caller.
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                values.append(value)
                continue
            if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
                values.append(int(value.strip(), 10))
                continue
            return None
        min_bs, max_bs, draft_tokens = values
        if min_bs < 1 or max_bs < min_bs or draft_tokens < 0:
            return None
        normalized.append((min_bs, max_bs, draft_tokens))

    normalized.sort(key=lambda item: (item[0], item[1], item[2]))
    for previous, current in zip(normalized, normalized[1:]):
        if current[0] <= previous[1]:
            return None
    return normalized


def parse_speculative_schedule(schedule_str: str) -> list[tuple[int, int, int]]:
    """Parse ``min_batch:max_batch:draft_tokens`` schedule entries.

    Invalid, empty, or ambiguous values safely fall back to a fresh copy of
    :data:`DEFAULT_SPECULATIVE_SCHEDULE`.  Returning a copy prevents a caller
    from mutating the process-wide default policy accidentally.
    """
    if not isinstance(schedule_str, str) or not schedule_str.strip():
        return list(DEFAULT_SPECULATIVE_SCHEDULE)

    entries: list[list[int]] = []
    try:
        for raw_entry in schedule_str.split(","):
            fields = [field.strip() for field in raw_entry.split(":")]
            if len(fields) != 3 or any(not field for field in fields):
                raise ValueError("each schedule entry must contain three integers")
            entries.append([int(field, 10) for field in fields])
    except (TypeError, ValueError, OverflowError):
        return list(DEFAULT_SPECULATIVE_SCHEDULE)

    normalized = _validated_speculative_schedule(entries)
    return normalized if normalized is not None else list(DEFAULT_SPECULATIVE_SCHEDULE)


def get_speculative_draft_tokens(
    batch_size: int,
    schedule: list | None = None,
) -> int:
    """Return the draft-token count for a scheduler batch size.

    ``schedule`` overrides :envvar:`VLLM_EXL3_SPEC_SCHEDULE`.  When neither is
    supplied, the default policy is 3/2/1 draft tokens for batches 1--4,
    5--8, and 9--16 respectively; all other batch sizes return zero.
    """
    try:
        batch_size = int(batch_size)
    except (TypeError, ValueError, OverflowError):
        return 0
    if batch_size <= 0:
        return 0

    if schedule is None:
        configured = os.environ.get(SPECULATIVE_SCHEDULE_ENV)
        active_schedule = (
            parse_speculative_schedule(configured)
            if configured is not None
            else list(DEFAULT_SPECULATIVE_SCHEDULE)
        )
    elif isinstance(schedule, str):
        active_schedule = parse_speculative_schedule(schedule)
    else:
        active_schedule = _validated_speculative_schedule(schedule)
        if active_schedule is None:
            active_schedule = list(DEFAULT_SPECULATIVE_SCHEDULE)

    for min_bs, max_bs, draft_tokens in active_schedule:
        if min_bs <= batch_size <= max_bs:
            return draft_tokens
    return 0


def is_adaptive_verification_enabled() -> bool:
    """Return whether confidence-based speculative verification is enabled.

    Only explicit affirmative values enable the feature.  This fail-closed
    policy keeps an unset, misspelled, or otherwise unknown environment value
    from changing verification behaviour unexpectedly in a serving process.
    """
    value = os.environ.get(ADAPTIVE_VERIFICATION_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def filter_speculative_candidates(
    probs: torch.Tensor,
    threshold: float = 0.5,
    *,
    return_tensor: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | int]:
    """Keep the confident prefix of each speculative candidate sequence.

    Candidate verification is sequential: once a candidate falls below
    ``threshold``, that candidate and every later candidate in the same
    sequence are pruned.  A one-dimensional input is treated as one sequence
    and returns a Python ``int`` count; batched inputs return one ``long``
    count per leading sequence. ``return_tensor=True`` keeps a one-dimensional
    result's count as a scalar tensor instead of synchronizing to a Python int.
    """
    if isinstance(threshold, bool):
        raise ValueError("threshold must be finite and in [0.0, 1.0]")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("threshold must be finite and in [0.0, 1.0]") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0.0, 1.0]")

    if not isinstance(probs, torch.Tensor):
        raise TypeError(f"probs must be a torch.Tensor, got {type(probs).__name__}")

    if probs.ndim == 0:
        raise ValueError("probs must have a candidate dimension (at least 1D)")

    num_candidates = probs.shape[-1]
    if num_candidates == 0:
        mask = torch.zeros_like(probs, dtype=torch.bool)
        if probs.ndim == 1:
            count = torch.zeros((), dtype=torch.long, device=probs.device)
            return mask, count if return_tensor else 0
        return mask, torch.zeros(probs.shape[:-1], dtype=torch.long, device=probs.device)

    confident = torch.ge(probs, threshold)
    # cumprod encodes the first-failure cutoff without Python loops or host
    # synchronization, so the operation remains on the candidate tensor's
    # device during decode.
    mask = torch.cumprod(confident.to(dtype=torch.int64), dim=-1).to(dtype=torch.bool)
    kept_counts = mask.sum(dim=-1, dtype=torch.long)
    if probs.ndim == 1:
        return mask, kept_counts if return_tensor else int(kept_counts.item())
    return mask, kept_counts


def _exl3_trellis_arena_enabled() -> bool:
    """Contiguous per-shape trellis arenas (default ON). Set 0 to use legacy allocs."""
    return os.environ.get("VLLM_EXL3_TRELLIS_ARENA", "1") != "0"


def _exl3_mem_waterfall_enabled() -> bool:
    return os.environ.get("VLLM_EXL3_MEM_WATERFALL", "0") == "1"


def _read_proc_meminfo_gib(*keys: str) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            wanted = set(keys)
            for line in fh:
                name, _, rest = line.partition(":")
                if name in wanted:
                    out[name] = int(rest.strip().split()[0]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return out


def _exl3_mem_snapshot(tag: str, layer: Any | None = None) -> dict[str, Any]:
    """Host + process + torch CUDA memory snapshot for materialization tracing."""
    snap: dict[str, Any] = {"tag": tag, "ts": time.time()}
    snap.update(_read_proc_meminfo_gib("MemAvailable", "AnonPages", "Cached"))
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    snap["VmRSS_GiB"] = int(line.split()[1]) / (1024.0 * 1024.0)
                elif line.startswith("VmSize:"):
                    snap["VmSize_GiB"] = int(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    try:
        with open("/proc/self/smaps_rollup", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("Pss:"):
                    snap["Pss_GiB"] = int(line.split()[1]) / (1024.0 * 1024.0)
                    break
    except OSError:
        pass
    if _TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
        try:
            snap["cuda_allocated_GiB"] = torch.cuda.memory_allocated() / (1024.0**3)
            snap["cuda_reserved_GiB"] = torch.cuda.memory_reserved() / (1024.0**3)
        except Exception:
            pass
    if layer is not None:
        snap["trellis_storage_count"] = _count_trellis_storages(layer)
        snap["trellis_final_bytes"] = _trellis_nbytes(layer)
        staging = getattr(layer, "_exl3_trellis_staging", None)
        if staging:
            snap["trellis_staging_bytes"] = sum(
                int(t.numel()) * int(t.element_size())
                for proj_map in staging.values()
                for t in proj_map.values()
                if t is not None
            )
    path = os.environ.get("VLLM_EXL3_MEM_WATERFALL_PATH", "")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(snap, sort_keys=True) + "\n")
        except OSError:
            pass
    return snap


def _count_trellis_storages(layer: Any) -> int:
    """Unique trellis backing storages (arena tensors when present)."""
    if not _TORCH_AVAILABLE or torch is None:
        return 0
    ptrs: set[int] = set()
    arena_lists = (
        getattr(layer, "_exl3_gate_trellis_arenas", None),
        getattr(layer, "_exl3_up_trellis_arenas", None),
        getattr(layer, "_exl3_down_trellis_arenas", None),
    )
    has_arenas = False
    for arenas in arena_lists:
        if not arenas:
            continue
        has_arenas = True
        for arena in arenas:
            try:
                ptrs.add(int(arena.untyped_storage().data_ptr()))
            except Exception:
                continue
    if has_arenas:
        return len(ptrs)
    for name in ("gate_trellis", "up_trellis", "down_trellis"):
        plist = getattr(layer, name, None)
        if plist is None:
            continue
        for p in plist:
            if p is None or int(getattr(p, "numel", lambda: 0)()) == 0:
                continue
            try:
                ptrs.add(int(p.untyped_storage().data_ptr()))
            except Exception:
                continue
    return len(ptrs)


def _trellis_nbytes(layer: Any) -> int:
    total = 0
    for name in ("gate_trellis", "up_trellis", "down_trellis"):
        plist = getattr(layer, name, None)
        if plist is None:
            continue
        for p in plist:
            if p is None or int(getattr(p, "numel", lambda: 0)()) == 0:
                continue
            total += int(p.numel()) * int(p.element_size())
    # Arenas may be counted twice if we also sum views; prefer arena bytes when present.
    arena_bytes = 0
    for aname in (
        "_exl3_gate_trellis_arenas",
        "_exl3_up_trellis_arenas",
        "_exl3_down_trellis_arenas",
    ):
        for arena in getattr(layer, aname, []) or []:
            arena_bytes += int(arena.numel()) * int(arena.element_size())
    return arena_bytes if arena_bytes else total


def _proj_from_shard_id(shard_id: str) -> str:
    if shard_id == "w1":
        return "gate"
    if shard_id == "w3":
        return "up"
    if shard_id == "w2":
        return "down"
    raise ValueError(f"unknown EXL3 shard_id={shard_id}")




# _PRESCAN_CACHE: one safetensors header parse per shard per process, not one
# safe_open per expert key (11,520 re-parses per rank otherwise). Shapes are
# immutable per (path, mtime, size).
_PRESCAN_CACHE: dict = {}


def _prescan_shape(model_dir: str, shard: str, key: str):
    path = os.path.join(model_dir, shard)
    try:
        st = os.stat(path)
        ck = (path, st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    tbl = _PRESCAN_CACHE.get(ck)
    if tbl is None:
        from safetensors import safe_open
        tbl = {}
        try:
            with safe_open(path, framework="pt") as f:
                for k in f.keys():
                    if k.endswith(".trellis"):
                        tbl[k] = tuple(int(x) for x in f.get_slice(k).get_shape())
        except Exception:
            return None
        _PRESCAN_CACHE[ck] = tbl
    return tbl.get(key)


def _try_prescan_trellis_shapes(
    layer: Any,
    num_experts: int,
) -> dict[str, dict[int, tuple[int, ...]]] | None:
    """Header-only shape scan from the on-disk checkpoint (no tensor materialize).

    Uses ``VLLM_ENGRAM_MODEL_DIR`` / ``VLLM_EXL3_MODEL_DIR`` and the layer's
    ``layer_name``/``prefix`` to locate ``layers.N.ffn.experts.*`` trellis keys.
    Local expert ids map linearly onto a global contiguous block when
    ``layer.starting_expert_offset`` / EP metadata is present; otherwise assume
    local id == global id (offline tests).
    """
    model_dir = os.environ.get("VLLM_ENGRAM_MODEL_DIR") or os.environ.get(
        "VLLM_EXL3_MODEL_DIR"
    )
    if not model_dir:
        return None
    try:
        from safetensors import safe_open
    except Exception:
        return None
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return None
    try:
        with open(index_path, encoding="utf-8") as fh:
            weight_map = json.load(fh).get("weight_map", {})
    except Exception:
        return None
    layer_name = str(
        getattr(layer, "layer_name", None)
        or getattr(layer, "prefix", None)
        or ""
    )
    # Expect ...layers.N.ffn.experts or layers.N
    import re as _re

    m = _re.search(r"layers\.(\d+)", layer_name)
    if not m:
        return None
    layer_id = int(m.group(1))
    offset = int(
        getattr(layer, "starting_expert_offset", None)
        or getattr(layer, "expert_id_offset", None)
        or 0
    )
    # EP linear placement: local e <-> global offset+e
    prefix = f"layers.{layer_id}.ffn.experts."
    shapes: dict[str, dict[int, tuple[int, ...]]] = {
        "gate": {},
        "up": {},
        "down": {},
    }
    proj_map = {"w1": "gate", "w3": "up", "w2": "down"}
    # Gather keys per local expert.
    for local_e in range(int(num_experts)):
        global_e = offset + local_e
        for wp, proj in proj_map.items():
            key = f"{prefix}{global_e}.{wp}.trellis"
            shard = weight_map.get(key)
            if shard is None:
                return None  # incomplete map; fall back to stage-pack
            path = os.path.join(model_dir, shard)
            from .tensor_metadata import current_tensor_metadata_provider
            provider = current_tensor_metadata_provider()
            if provider is not None:
                # Authoritative metadata is supplied before construction by the
                # selected loader. Failure must propagate, never silently stage.
                desc = provider(path, key)
                shape = tuple(desc.shape)
                if (desc.dtype != "I16" or len(shape) != 3
                        or any(type(x) is not int or x <= 0 for x in shape)
                        or shape[-1] % 16 or not 2 <= shape[-1] // 16 <= 8
                        or math.prod(shape) * 2 != desc.nbytes):
                    raise ValueError("invalid EXL3 planned tensor metadata: " + key)
            else:
                shape = _prescan_shape(model_dir, shard, key)  # _PRESCAN_CACHE
                if shape is None:
                    return None
            shapes[proj][local_e] = shape
    return shapes


def _uva_trellis_placement_requested(layer: Any) -> bool:
    """True when the packed routed-expert payload must live in pinned host memory.

    Either the recipe demanded it (VLLM_EXL3_REQUIRE_UVA_EXPERTS=1) or vLLM's
    UVA offloader already marked the layer's trellis placeholders at
    construction time. In both cases the real payload allocated here must not
    silently land on the accelerator.
    """
    from .uva_offload import uva_expert_offload_required

    if uva_expert_offload_required():
        return True
    placeholder = getattr(layer, "w13_trellis", None)
    return bool(getattr(placeholder, "_vllm_is_uva_offloaded", False))


def _pinned_host_empty(shape: tuple[int, ...], dtype: "torch.dtype") -> "torch.Tensor":
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)


def _alloc_trellis_arena(
    layer: Any, shape: tuple[int, ...], dest_device: "torch.device"
) -> "torch.Tensor":
    """Allocate one trellis arena on ``dest_device`` or, for UVA runs, as an
    accelerator view of pinned host memory (vLLM's zero-copy placement)."""
    if not _uva_trellis_placement_requested(layer):
        return torch.empty(shape, dtype=torch.int16, device=dest_device)
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    host = _pinned_host_empty(tuple(int(x) for x in shape), torch.int16)
    view = get_accelerator_view_from_cpu_tensor(host)
    # Keep the pinned storage alive for as long as the layer exists.
    keep = layer.__dict__.setdefault("_exl3_uva_host_arenas", [])
    keep.append(host)
    view._vllm_is_uva_offloaded = True
    return view


def _arena_parameter(arena: "torch.Tensor") -> "Parameter":
    p = Parameter(arena, requires_grad=False)
    if getattr(arena, "_vllm_is_uva_offloaded", False):
        p._vllm_is_uva_offloaded = True
    return p


def prepare_trellis_arena_plan(
    layer: Any,
    shapes_by_proj: dict[str, dict[int, tuple[int, ...]]],
) -> dict[str, Any]:
    """Pre-allocate contiguous per-shape arenas and map expert_id -> slot.

    ``shapes_by_proj`` maps proj in {gate,up,down} -> {expert_id: exact_shape}.
    Subsequent ``_load_exl3`` trellis loads copy directly into the planned slot
    (safetensors -> FINAL) without retaining a full-layer staging set.
    """
    if not _TORCH_AVAILABLE or torch is None:
        raise RuntimeError("torch required for trellis arenas")
    dest_device = layer.w13_suh.device
    proj_to_plist = {
        "gate": layer.gate_trellis,
        "up": layer.up_trellis,
        "down": layer.down_trellis,
    }
    proj_to_attr = {
        "gate": "_exl3_gate_trellis_arenas",
        "up": "_exl3_up_trellis_arenas",
        "down": "_exl3_down_trellis_arenas",
    }
    plan: dict[str, dict[tuple[int, ...], dict[str, Any]]] = {}
    eid_index: dict[str, dict[int, tuple[tuple[int, ...], int]]] = {
        "gate": {},
        "up": {},
        "down": {},
    }
    stats: dict[str, Any] = {
        "planned": True,
        "arenas": {},
        "allocations_after": 0,
        "final_bytes": 0,
        "temp_peak_bytes": 0,
    }
    for proj, plist in proj_to_plist.items():
        by_shape: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for eid, shape in sorted((shapes_by_proj.get(proj) or {}).items()):
            by_shape[tuple(int(x) for x in shape)].append(int(eid))
        arenas: list[Parameter] = []
        plan[proj] = {}
        n_experts = int(len(plist))
        for shape, eids in by_shape.items():
            n = len(eids)
            arena = _alloc_trellis_arena(layer, (n, *shape), dest_device)
            meta = {
                "arena": arena,
                "eid_to_idx": {eid: i for i, eid in enumerate(eids)},
            }
            plan[proj][shape] = meta
            for i, eid in enumerate(eids):
                if not (0 <= eid < n_experts):
                    raise RuntimeError(f"EXL3 arena plan expert out of range: {eid}")
                view = arena[i]
                new_p = Parameter(view, requires_grad=False)
                new_p.weight_loader = getattr(plist[eid], "weight_loader", None)
                new_p._exl3_owner = layer
                plist[eid] = new_p
                eid_index[proj][eid] = (shape, i)
            arenas.append(_arena_parameter(arena))
            stats["arenas"].setdefault(proj, []).append(
                {"shape": list(shape), "n": n, "bytes": int(arena.nbytes)}
            )
            stats["allocations_after"] += 1
            stats["final_bytes"] += int(arena.nbytes)
        setattr(layer, proj_to_attr[proj], arenas)
    layer._exl3_trellis_arena_plan = plan
    layer._exl3_trellis_eid_index = eid_index
    layer._exl3_trellis_staging = {"gate": {}, "up": {}, "down": {}}
    layer._exl3_trellis_arena_stats = stats
    layer._exl3_trellis_temp_peak_bytes = 0
    return stats


# Process-wide direct-fill counters (prove the real load hits this path).
_DIRECT_FILL_STATS = {
    "DIRECT_FILL_CALLS": 0,
    "DIRECT_FILL_BYTES": 0,
    "DIRECT_FILL_FALLBACK_CALLS": 0,
    "DIRECT_FILL_FALLBACK_BYTES": 0,
    "DIRECT_FILL_DEVICE": "",
    "MADV_AFTER_H2D_CALLS": 0,
    "MADV_AFTER_H2D_BYTES": 0,
}


def direct_fill_stats() -> dict[str, Any]:
    """Return current direct fill call count and byte transfer volume."""
    return dict(_DIRECT_FILL_STATS)


def _find_containing_vma(addr: int) -> tuple[int, int, str] | None:
    """Return (vma_start, vma_end, pathname) for addr from /proc/self/maps."""
    try:
        with open("/proc/self/maps", "r", encoding="utf-8") as fh:
            for line in fh:
                # e.g. 7f..-7f.. rw-p 00000000 00:00 0  [/path]
                parts = line.split()
                if not parts:
                    continue
                span = parts[0]
                if "-" not in span:
                    continue
                lo_s, hi_s = span.split("-", 1)
                lo, hi = int(lo_s, 16), int(hi_s, 16)
                if lo <= addr < hi:
                    path = parts[-1] if len(parts) >= 6 and parts[-1].startswith("/") else ""
                    return lo, hi, path
    except Exception:
        return None
    return None


def _madv_dontneed_cpu_tensor(src: "torch.Tensor") -> bool:
    """Advise kernel to drop COW pages of a *consumed tensor view* after H2D.

    Range is the VIEW byte span (``data_ptr`` + ``numel*element_size``), never
    the base ``untyped_storage()`` span — advising the storage can discard
    pages of later unconsumed views of a fused shard (P1).

    Further bounded to the containing VMA, and only applied for file-backed
    ``*.safetensors`` mappings (the MAP_PRIVATE COW case). Non-contiguous
    tensors and heap mappings are skipped. Toggle off with
    ``VLLM_EXL3_MADV_AFTER_H2D=0``.
    """
    if os.environ.get("VLLM_EXL3_MADV_AFTER_H2D", "1") == "0":
        return False
    if not _TORCH_AVAILABLE or torch is None:
        return False
    if not torch.is_tensor(src) or src.device.type != "cpu" or src.numel() == 0:
        return False
    # Do not advise spans that contain gaps between elements.
    if not src.is_contiguous():
        return False
    try:
        import ctypes

        view_start = int(src.data_ptr())
        view_nbytes = int(src.numel()) * int(src.element_size())
        if view_start == 0 or view_nbytes <= 0:
            return False
        view_end = view_start + view_nbytes
        vma = _find_containing_vma(view_start)
        if vma is None:
            return False
        vma_lo, vma_hi, vma_path = vma
        # Only reclaim safetensors MAP_PRIVATE file pages — not arbitrary heap.
        if not vma_path.endswith(".safetensors"):
            return False
        safe_start = max(view_start, vma_lo)
        safe_end = min(view_end, vma_hi)
        page = os.sysconf("SC_PAGESIZE")
        # Page-align INWARD only — never touch adjacent views / heap.
        start = safe_start + ((page - (safe_start % page)) % page)
        end = safe_end - (safe_end % page)
        if end <= start:
            return False
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libc.madvise.restype = ctypes.c_int
        advised = end - start
        rc = libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(advised), 4)
        if rc == 0:
            _DIRECT_FILL_STATS["MADV_AFTER_H2D_CALLS"] += 1
            _DIRECT_FILL_STATS["MADV_AFTER_H2D_BYTES"] += advised
            return True
    except Exception:
        return False
    return False



# _P1_POPULATE: install PTEs for the mmap'd safetensors view on the CPU side before the
# H2D copy. Source views are MAP_PRIVATE file pages (safe_open pt backend); without this
# the copy takes one fault per 4 KiB page (~80 MB/s observed on GB10).
_P1_LIBC = None
_P1_MODE = os.environ.get("VLLM_EXL3_PREFETCH", "populate")   # populate | willneed | clone | off
_P1_LOOKAHEAD = int(os.environ.get("VLLM_EXL3_PREFETCH_LOOKAHEAD_MB", "128")) << 20
_P1_STATS = {"calls": 0, "bytes": 0, "secs": 0.0, "errno": 0, "clone_fallbacks": 0}
_MADV_WILLNEED, _MADV_POPULATE_READ = 3, 22


def _p1_libc():
    global _P1_LIBC
    if _P1_LIBC is None:
        import ctypes
        lib = ctypes.CDLL("libc.so.6", use_errno=True)
        lib.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        lib.madvise.restype = ctypes.c_int
        _P1_LIBC = lib
    return _P1_LIBC


def _p1_prefault(src: "torch.Tensor") -> "torch.Tensor":
    """Return a tensor whose pages are resident: ``src`` after MADV_POPULATE_READ
    (+ MADV_WILLNEED lookahead into the following bytes of the same mapping), or a
    clone() if madvise is refused (kernel < 5.14) / mode=clone."""
    if _P1_MODE == "off" or src.device.type != "cpu" or not src.is_contiguous():
        return src
    n = int(src.numel()) * int(src.element_size())
    if n == 0:
        return src
    import ctypes
    page = 4096
    start = int(src.data_ptr()) & ~(page - 1)            # round DOWN: always inside the file mapping
    end = (int(src.data_ptr()) + n) & ~(page - 1)        # round DOWN: never past the mapping end
    t0 = time.monotonic()
    if _P1_MODE in ("populate", "willneed") and end > start:
        lib = _p1_libc()
        adv = _MADV_POPULATE_READ if _P1_MODE == "populate" else _MADV_WILLNEED
        rc = lib.madvise(ctypes.c_void_p(start), ctypes.c_size_t(end - start), adv)
        if rc == 0:
            if _P1_LOOKAHEAD > 0:
                # async readahead of what follows this view (safetensors lays data out in key
                # order, and keys are iterated sorted). ENOMEM past the mapping end is harmless.
                lib.madvise(ctypes.c_void_p(end), ctypes.c_size_t(_P1_LOOKAHEAD), _MADV_WILLNEED)
            _P1_STATS["calls"] += 1; _P1_STATS["bytes"] += end - start
            _P1_STATS["secs"] += time.monotonic() - t0
            return src
        _P1_STATS["errno"] = ctypes.get_errno()
    out = src.clone()                                    # CPU memcpy -> CPU faults w/ fault-around + readahead
    _P1_STATS["clone_fallbacks"] += 1
    _P1_STATS["secs"] += time.monotonic() - t0
    return out




# _BOUNCE2: never let the GB10 copy engine translate file-backed pageable pages
# through ATS (~250 MB/s measured). CPU memcpy into a pinned bounce first
# (5-27 GB/s), then async H2D. Two buffers, one per slot: one buffer is
# overwritten by the next memcpy while the previous H2D is still reading it.
# Gate: direct 201/251 MB/s cold/warm vs bounce 5.1/27.2 GB/s cold/warm
# on shard-05 trellis views.
_BOUNCE_STATE = {"bufs": [None, None], "evt": [None, None], "slot": 0, "cap": 0}


def _bounce_copy(dst: "torch.Tensor", src: "torch.Tensor") -> None:
    """Pinned H2D bounce. Two buffers, one per slot.

    A single pinned buffer is a race: the next CPU memcpy overwrites it
    while the previous non-blocking H2D is still reading it. Each slot has
    its own buffer, and the slot event is waited before that buffer is reused.
    Growing the buffers waits for both in-flight copies first.
    """
    n = int(src.numel()) * int(src.element_size())
    st = _BOUNCE_STATE
    if st["bufs"][0] is None or st["cap"] < n:
        for evt in st["evt"]:
            if evt is not None:
                evt.synchronize()
        try:
            bufs = [
                torch.empty(n, dtype=torch.uint8, device="cpu", pin_memory=True),
                torch.empty(n, dtype=torch.uint8, device="cpu", pin_memory=True),
            ]
            evts = [torch.cuda.Event(), torch.cuda.Event()]
        except (RuntimeError, AssertionError):
            # No pinned allocator (CPU-only torch, e.g. the unit-test runner).
            # The bounce exists to keep the GB10 copy engine off file-backed
            # pageable pages; with no accelerator there is nothing to bounce
            # for, so fall back to the direct blocking copy.
            st["bufs"] = [None, None]
            st["evt"] = [None, None]
            st["cap"] = 0
            dst.copy_(src, non_blocking=False)
            return
        st["bufs"] = bufs
        st["evt"] = evts
        st["cap"] = n
        st["slot"] = 0
    i = st["slot"]
    st["evt"][i].synchronize()  # previous H2D from this slot's buffer retired
    bv = st["bufs"][i][:n].view(src.dtype).view(src.shape)
    bv.copy_(src)  # CPU memcpy: page cache -> this slot's pinned buffer
    dst.copy_(bv, non_blocking=True)
    st["evt"][i].record()
    st["slot"] = i ^ 1


def _direct_fill_trellis_slot(
    layer: Any,
    proj: str,
    expert_id: int,
    src: "torch.Tensor",
) -> None:
    """Copy one trellis into its pre-planned arena slot; drop ``src`` ASAP."""
    eid_index = getattr(layer, "_exl3_trellis_eid_index", None)
    plan = getattr(layer, "_exl3_trellis_arena_plan", None)
    if not eid_index or not plan:
        raise RuntimeError("EXL3 direct fill requires prepare_trellis_arena_plan")
    if expert_id not in eid_index[proj]:
        raise RuntimeError(
            f"EXL3 arena plan missing {proj} expert={expert_id} shape={tuple(src.shape)}"
        )
    shape, idx = eid_index[proj][expert_id]
    if tuple(int(x) for x in src.shape) != shape:
        raise RuntimeError(
            f"EXL3 arena slot shape mismatch {proj} expert={expert_id}: "
            f"got {tuple(src.shape)} planned {shape}"
        )
    arena = plan[proj][shape]["arena"]
    transient = int(src.numel()) * int(src.element_size())
    layer._exl3_trellis_temp_peak_bytes = max(
        int(getattr(layer, "_exl3_trellis_temp_peak_bytes", 0)), transient
    )
    # PR14's direct H2D copy, kept separate from its broader policy changes.
    # Keep conversion on the source device; never allocate src.to(cuda) beside
    # the final arena. Blocking copy establishes completion before release.
    if src.dtype != torch.int16:
        src = src.to(dtype=torch.int16)
    if not src.is_contiguous():
        src = src.contiguous()
    src = _p1_prefault(src)  # _P1_POPULATE: install PTEs before the CPU memcpy
    if arena[idx].is_cuda:
        # Pinned bounce: one host memcpy plus an async H2D, no per-tensor sync.
        _bounce_copy(arena[idx], src)
    else:
        # CPU arena (unit tests, CPU-only torch): blocking copy establishes
        # completion before ``src`` is released.
        arena[idx].copy_(src, non_blocking=False)
    _madv_dontneed_cpu_tensor(src)
    _DIRECT_FILL_STATS["DIRECT_FILL_CALLS"] += 1
    _DIRECT_FILL_STATS["DIRECT_FILL_BYTES"] += transient
    _DIRECT_FILL_STATS["DIRECT_FILL_DEVICE"] = str(arena.device)


def _pack_trellis_arenas(layer: Any) -> dict[str, Any]:
    """Pack staged per-expert trellis tensors into contiguous per-shape arenas.

    Each expert keeps an exact-shape view (no padding/truncation). Heterogeneous
    K is preserved by grouping only equal shapes into the same arena.

    Prefer ``prepare_trellis_arena_plan`` + direct fill to avoid holding a full
    staging set beside the final arenas (UMA source+dest coexistence).
    """
    if not _TORCH_AVAILABLE or torch is None:
        raise RuntimeError("torch required for trellis arenas")
    # Already planned+filled: just report stats.
    if getattr(layer, "_exl3_trellis_arena_plan", None) is not None:
        stats = dict(getattr(layer, "_exl3_trellis_arena_stats", {}) or {})
        stats["packed"] = True
        stats["mode"] = "direct_plan"
        stats["temp_peak_bytes"] = int(
            getattr(layer, "_exl3_trellis_temp_peak_bytes", 0)
        )
        layer._exl3_trellis_arena_stats = stats
        return stats

    staging: dict[str, dict[int, torch.Tensor]] = getattr(
        layer, "_exl3_trellis_staging", None
    ) or {}
    if not staging:
        return {"packed": False, "reason": "no_staging"}

    dest_device = layer.w13_suh.device
    stats: dict[str, Any] = {
        "packed": True,
        "mode": "post_stage_pack",
        "arenas": {},
        "allocations_after": 0,
        "final_bytes": 0,
        "temp_peak_bytes": 0,
    }
    temp_bytes = 0
    for proj_map in staging.values():
        for t in proj_map.values():
            temp_bytes += int(t.numel()) * int(t.element_size())
    stats["temp_peak_bytes"] = temp_bytes

    proj_to_plist = {
        "gate": layer.gate_trellis,
        "up": layer.up_trellis,
        "down": layer.down_trellis,
    }
    proj_to_attr = {
        "gate": "_exl3_gate_trellis_arenas",
        "up": "_exl3_up_trellis_arenas",
        "down": "_exl3_down_trellis_arenas",
    }

    for proj, plist in proj_to_plist.items():
        by_shape: dict[tuple[int, ...], list[tuple[int, torch.Tensor]]] = defaultdict(
            list
        )
        for eid, tensor in sorted((staging.get(proj) or {}).items()):
            if tensor is None or int(tensor.numel()) == 0:
                continue
            if tensor.dtype != torch.int16:
                tensor = tensor.to(dtype=torch.int16)
            shape = tuple(int(x) for x in tensor.shape)
            by_shape[shape].append((int(eid), tensor))

        arenas: list[Parameter] = []
        n_experts = int(len(plist))
        for shape, items in by_shape.items():
            n = len(items)
            arena = _alloc_trellis_arena(layer, (n, *shape), dest_device)
            for i, (eid, src) in enumerate(items):
                if not (0 <= eid < n_experts):
                    raise RuntimeError(f"EXL3 arena expert id out of range: {eid}")
                if src.device == dest_device and src.dtype == torch.int16:
                    arena[i].copy_(src if src.is_contiguous() else src.contiguous())
                else:
                    arena[i].copy_(
                        src.to(device=dest_device, dtype=torch.int16, non_blocking=False)
                    )
                view = arena[i]
                new_p = Parameter(view, requires_grad=False)
                new_p.weight_loader = getattr(plist[eid], "weight_loader", None)
                new_p._exl3_owner = layer
                plist[eid] = new_p
                staging[proj].pop(eid, None)
                del src
            arenas.append(_arena_parameter(arena))
            stats["arenas"].setdefault(proj, []).append(
                {"shape": list(shape), "n": n, "bytes": int(arena.nbytes)}
            )
            stats["allocations_after"] += 1
            stats["final_bytes"] += int(arena.nbytes)
            # Free host pages for this shape group before the next alloc on UMA.
            if os.environ.get("EXL3_STAGING_GC", "0") == "1":
                gc.collect()
        setattr(layer, proj_to_attr[proj], arenas)

    layer._exl3_trellis_staging = {"gate": {}, "up": {}, "down": {}}
    if os.environ.get("EXL3_STAGING_GC", "0") == "1":
        gc.collect()
    if torch.cuda.is_available() and os.environ.get("EXL3_STAGING_GC", "0") == "1":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return stats


def _moe_tp_align() -> int:
    """Column alignment for routed-expert TP shards (VLLM_EXL3_MOE_TP_ALIGN, 0 = off).

    EXL3 applies its input/output Hadamard transforms in 128-wide blocks, so a
    TP shard boundary must not cut a block. With 128 set, the intermediate dim
    is split in whole blocks, unevenly when needed: DSV4.1's 2304 over tp=4
    becomes 640/640/512/512 instead of the 576 equal chunk that straddles blocks.
    """
    try:
        return int(os.environ.get("VLLM_EXL3_MOE_TP_ALIGN", "0"))
    except ValueError:
        return 0


def aligned_tp_split(size: int, tp_rank: int, tp_size: int, align: int) -> tuple[int, int]:
    """(offset, length) of rank's shard when ``size`` is split in ``align`` blocks."""
    if size % align:
        raise ValueError(f"EXL3 aligned TP shard: size {size} is not a multiple of {align}")
    base, rem = divmod(size // align, tp_size)
    counts = [base + (1 if r < rem else 0) for r in range(tp_size)]
    return sum(counts[:tp_rank]) * align, counts[tp_rank] * align


def moe_tp_rotation(layer: Any, tp_size: int) -> int:
    """Per-layer chunk rotation for aligned MoE TP (VLLM_EXL3_MOE_TP_ROTATE=1, 0 = off).

    An uneven aligned split (2304 over tp=4 -> 640/640/512/512) always gives the
    larger chunks to the same ranks, a ~13 GiB weight imbalance on DSV4.1. With
    rotation, rank r of layer L takes chunk (r + L) % tp_size. Every rank still
    computes a partial sum over its own slice and the MoE all-reduce adds them,
    so the output is unchanged; only which rank holds which slice moves.
    """
    if os.environ.get("VLLM_EXL3_MOE_TP_ROTATE", "0") != "1" or tp_size <= 1:
        return 0
    name = str(getattr(layer, "layer_name", None) or getattr(layer, "prefix", None) or "")
    m = re.search(r"layers\.(\d+)", name)
    return int(m.group(1)) % tp_size if m else 0


def _narrow_tp(
    tensor: torch.Tensor,
    dim: int,
    tp_rank: int,
    tp_size: int,
    unit: int = 1,
    aligned: bool = False,
) -> torch.Tensor:
    """Narrow ``dim`` to this rank's shard.

    ``aligned`` (routed experts only) applies the VLLM_EXL3_MOE_TP_ALIGN block
    split; ``unit`` is columns per index along ``dim`` (16 for trellis tiles).
    """
    if tp_size <= 1:
        return tensor
    size = int(tensor.shape[dim])
    align = _moe_tp_align() if aligned else 0
    if align > 0:
        if (size * unit) % align:
            # Falling through to the equal split here would cut a Hadamard block
            # and decode every shard against the wrong transform, silently. This
            # path is only reachable with the opt-in env var set, so refuse the
            # geometry instead of loading weights that cannot be right.
            raise RuntimeError(
                f"EXL3 aligned MoE TP: dim {dim} spans {size * unit} columns, which "
                f"is not a multiple of the VLLM_EXL3_MOE_TP_ALIGN={align} Hadamard "
                "block; unset VLLM_EXL3_MOE_TP_ALIGN or use an expert-parallel build"
            )
        off, length = aligned_tp_split(size * unit, tp_rank, tp_size, align)
        return tensor.narrow(dim, off // unit, length // unit).contiguous()
    if size % tp_size:
        raise ValueError(
            f"EXL3 TP shard: dim {dim} size {size} is not divisible by tp={tp_size}"
        )
    chunk = size // tp_size
    return tensor.narrow(dim, chunk * tp_rank, chunk).contiguous()


def _resolve_tp_geometry(*owners: Any) -> tuple[int, int]:
    """Resolve per-layer TP metadata before consulting process-wide TP state.

    Rank and size resolve INDEPENDENTLY: a module may expose one without the
    other (VocabParallelEmbedding stores tp_size but not tp_rank). Coupling
    them made the resolver return rank 0 with the module's size — every rank
    then loaded shard 0 (the lm_head constant-token bug).
    """
    for owner in owners:
        if owner is None:
            continue
        rank = getattr(owner, "tp_rank", None)
        size = getattr(owner, "moe_tp_size", None)
        if size is None:
            size = getattr(owner, "tp_size", None)
        if size is None:
            size = getattr(owner, "_exl3_tp_size", None)
        if rank is not None or size is not None:
            resolved_rank = int(rank) if rank is not None else None
            resolved_size = int(size) if size is not None else None
            if resolved_rank is None or resolved_size is None:
                from vllm.distributed import (
                    get_tensor_model_parallel_rank,
                    get_tensor_model_parallel_world_size,
                )

                if resolved_rank is None:
                    resolved_rank = get_tensor_model_parallel_rank()
                if resolved_size is None:
                    resolved_size = get_tensor_model_parallel_world_size()
            return resolved_rank, resolved_size

    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()


def shard_exl3_col(
    loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int, aligned: bool = False
) -> torch.Tensor:
    """Gate/up: trellis dim 1 and svh dim 0 are column-parallel."""
    if suffix == "trellis":
        return _narrow_tp(loaded, 1, tp_rank, tp_size, 16, aligned)
    if suffix == "svh":
        return _narrow_tp(loaded, 0, tp_rank, tp_size, 1, aligned)
    return loaded.contiguous()


def shard_exl3_row(
    loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int, aligned: bool = False
) -> torch.Tensor:
    """Down: trellis dim 0 and suh dim 0 are row-parallel."""
    if suffix == "trellis":
        return _narrow_tp(loaded, 0, tp_rank, tp_size, 16, aligned)
    if suffix == "suh":
        return _narrow_tp(loaded, 0, tp_rank, tp_size, 1, aligned)
    return loaded.contiguous()


def _install_exllamav3_namespace() -> None:
    """Validate that the native ExLlamaV3 package and extension are importable."""
    import exllamav3_ext  # noqa: F401  — compiled extension must exist

    # ExLlamaV3 1.4+ imports cleanly as a regular package and its LinearEXL3
    # constructor relies on the real NullConfig/InferParams implementation.
    # Namespace stubs used by much older builds hide those classes and fail only
    # after a full checkpoint load, so deliberately exercise the normal import.
    importlib.import_module("exllamav3.modules.quant.exl3")


def load_linear_exl3_cls():
    _install_exllamav3_namespace()
    return importlib.import_module("exllamav3.modules.quant.exl3").LinearEXL3


def make_linear_exl3(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: torch.Tensor | None = None,
    mul1: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype | None = None,
):
    """Build a LinearEXL3 over already-sharded packed tensors. No BF16 expand."""
    if out_dtype is None and torch is not None:
        out_dtype = torch.float16
    cls = load_linear_exl3_cls()
    return cls(
        config=None,
        in_features=int(suh.numel()),
        out_features=int(svh.numel()),
        trellis=trellis.contiguous(),
        suh=suh.contiguous(),
        svh=svh.contiguous(),
        mcg=mcg.contiguous() if mcg is not None else None,
        mul1=mul1.contiguous() if mul1 is not None else None,
        out_dtype=out_dtype,
        transformers_fix=True,
    )


def execute_exl3_linear(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: torch.Tensor | None = None,
    mul1: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Real EXL3 expert GEMM entry (LinearEXL3 / exllamav3_ext)."""
    if out_dtype is None and torch is not None:
        out_dtype = torch.float32
    inner = make_linear_exl3(
        trellis, suh, svh, mcg, mul1,
        out_dtype=torch.float16 if torch is not None else None,
    )
    return inner.forward(x.contiguous().half(), {}, out_dtype=out_dtype)


def fused_moe_enabled() -> bool:
    return os.environ.get("EXL3_FUSED_MOE", "1") != "0"


def load_exllamav3_ext():
    import exllamav3_ext

    return exllamav3_ext


def _load_native_exl3_ext():
    """Return the optional native extension, without making it a hard dependency."""
    try:
        module = importlib.import_module("vllm_exl3_c")
    except Exception:
        return None
    return module if callable(getattr(module, "p2b_fused_moe", None)) else None


def native_moe_kernel_available() -> bool:
    """Whether the compiled cooperative native MoE entry point is available."""
    return _load_native_exl3_ext() is not None


def _exllamav3_moe_available() -> bool:
    try:
        return callable(getattr(load_exllamav3_ext(), "exl3_moe", None))
    except Exception:
        return False


def get_moe_kernel_backend() -> str:
    """Resolve ``VLLM_EXL3_MOE_KERNEL`` to the requested/available backend.

    ``native`` and ``exllamav3`` are intentionally returned as requested even
    when their optional extension is absent.  The dispatch function then applies
    the documented graceful fallback; this makes configuration introspection
    deterministic and avoids importing CUDA extensions during config parsing.
    """
    requested = os.environ.get("VLLM_EXL3_MOE_KERNEL", "auto").strip().lower()
    if requested not in {"native", "exllamav3", "auto"}:
        logger.warning(
            "Unknown VLLM_EXL3_MOE_KERNEL=%r; using auto selection", requested
        )
        requested = "auto"
    if requested != "auto":
        return requested
    if native_moe_kernel_available():
        return "native"
    if _exllamav3_moe_available():
        return "exllamav3"
    return "loop"


def _exl3_moe_accepts_num_active(fn) -> bool:
    try:
        import inspect

        if "num_active" in inspect.signature(fn).parameters:
            return True
    except (TypeError, ValueError):
        pass
    doc = getattr(fn, "__doc__", None) or ""
    return "num_active" in doc or "arg29" in doc or doc.count("arg") >= 30


# Positional arity of exllamav3's exl3_moe binding by release. The binding has no
# parameter names, so the pybind docstring ("arg0: ..., arg34: ...") is the contract.
EXL3_MOE_ARITY_147 = 30  # ..., act_limit, num_active            (exllamav3 <= 1.4.x)
EXL3_MOE_ARITY_150 = 35  # + output_scratch, fused_base, count_lo, count_hi, m_tile (>= 1.5.0)


def _exl3_moe_arity(fn) -> int | None:
    """Number of positional arguments the bound exl3_moe takes, or None if unreadable."""
    import re

    doc = getattr(fn, "__doc__", None) or ""
    idx = [int(m) for m in re.findall(r"\barg(\d+)\s*:", doc)]
    if idx:
        return max(idx) + 1
    try:
        import inspect

        params = inspect.signature(fn).parameters.values()
        if any(p.kind == p.VAR_POSITIONAL for p in params):
            return None
        return len(params)
    except (TypeError, ValueError):
        return None


def _exl3_moe_temp_rows(temps) -> int:
    """Row capacity of the fused temp buffers ([concurrency, rows, width]); the module
    default when a caller hands in placeholders instead of tensors."""
    first = temps[0] if temps else None
    shape = getattr(first, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[-2])
    return int(TEMP_ROWS_FUSED)


def _exl3_moe_tail(fn, temp_rows: int) -> tuple:
    """Trailing arguments exllamav3 1.5.0 added to exl3_moe, or () for older bindings.

    1.5.0 appended output_scratch and fused_base (fp32 slot scratch plus slot table for
    its deterministic-accumulation mode; None keeps the atomic scatter-add this plugin
    relies on), count_lo and count_hi (the per-expert row-count band this launch owns,
    1..temp rows covers every expert the fused kernel can take) and m_tile (kernel row
    tile; 16 is the only instance the pre-1.5.0 kernel had). These values reproduce the
    1.4.x all-fused launch, so the plugin's dispatch, fat-expert cap and temp buffers are
    unchanged. Measured on one GB10 with Qwen3.8-Flash-Next 3.05 bpw: 52.05 tok/s at MTP
    k=3 on 1.5.0 against 52.22 on 1.4.7, 28.54 against 27.77 without a draft.
    """
    arity = _exl3_moe_arity(fn)
    if arity is not None and arity >= EXL3_MOE_ARITY_150:
        return (None, None, 1, int(temp_rows), 16)
    return ()


def pin_exl3_expert_map(
    layer: torch.nn.Module, device: torch.device
) -> torch.Tensor | None:
    """Move expert_map onto `device` once. CUDA graph capture forbids a CPU→GPU copy."""
    emap = getattr(layer, "expert_map", None)
    if emap is None:
        return None
    raw_id = id(emap)
    cached = getattr(layer, "_exl3_pinned_expert_map", None)
    if (
        getattr(layer, "_exl3_raw_expert_map_id", None) == raw_id
        and cached is not None
        and cached.device == device
        and cached.dtype == torch.long
    ):
        return cached
    pinned = emap.to(device=device, dtype=torch.long)
    layer._exl3_pinned_expert_map = pinned
    layer._exl3_raw_expert_map_id = raw_id
    return pinned


def map_topk_to_local(
    ids: torch.Tensor,
    n_local: int,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    """ids (T, K) global expert ids → local ids, invalid/non-local → n_local sentinel.

    `expert_map` must already live on `ids.device` (see pin_exl3_expert_map).
    """
    flat = ids.reshape(-1)
    if expert_map is None:
        invalid = (flat < 0) | (flat >= n_local)
        return torch.where(invalid, flat.new_full(flat.shape, n_local), flat)
    if expert_map.device != flat.device or expert_map.dtype != torch.long:
        raise RuntimeError(
            "EXL3 expert_map is not pinned to the hidden-state device; "
            "call pin_exl3_expert_map before fused apply (CUDA graphs forbid the copy)"
        )
    n_global = int(expert_map.numel())
    safe = flat.clamp(min=0, max=max(n_global - 1, 0))
    mapped = expert_map[safe] if n_global else flat.new_full(flat.shape, n_local)
    invalid = (flat < 0) | (flat >= n_global) | (mapped < 0) | (mapped >= n_local)
    return torch.where(invalid, flat.new_full(flat.shape, n_local), mapped)


def apply_exl3_python_loop(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float | None = None,
    *,
    only_experts: set[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unique-expert LinearEXL3 loop. `only_experts` is local ids (fat-expert fallback)."""
    tokens, hidden = x2d.shape
    if out is None:
        out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x2d.device)
    local_ids = map_topk_to_local(ids, len(inners), expert_map).reshape_as(ids)
    for raw in torch.unique(local_ids).tolist():
        e = int(raw)
        if e >= len(inners):
            continue
        if only_experts is not None and e not in only_experts:
            continue
        token_idx, k_pos = (local_ids == e).nonzero(as_tuple=True)
        h = x2d.index_select(0, token_idx)
        pack = inners[e]
        gate = pack["gate"].forward(h.contiguous().half(), {}, out_dtype=torch.float32)
        up = pack["up"].forward(h.contiguous().half(), {}, out_dtype=torch.float32)
        if limit is not None and limit > 0:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        act = F.silu(gate) * up
        act_mask = pack.get("act_mask")
        if act_mask is not None:
            act.mul_(act_mask)
        down = pack["down"].forward(act.contiguous().half(), {}, out_dtype=torch.float32)
        scale = weights[token_idx, k_pos].unsqueeze(-1).to(dtype=torch.float32)
        out.index_add_(0, token_idx, down * scale)
    return out


def build_exl3_fused_state(layer: torch.nn.Module, inners: list[dict[str, Any]]) -> None:
    """Pointer tables + fused temps, once after load. No per-token alloc."""
    mixed_k = getattr(layer, "_exl3_mixed_k", False)
    if getattr(layer, "_exl3_mixed_store", None) is not None and not mixed_k:
        raise ValueError("EXL3 mixed-K cannot build uniform-K fused pointer tables")
    try:
        exllamav3_ext = load_exllamav3_ext()
    except Exception:
        exllamav3_ext = None

    device = layer.w13_suh.device
    n_exp = len(inners)
    # Gate/up input rotations are immutable after load. Cache this compatibility
    # fact once so fat-prefill dispatch never calls torch.equal on CUDA tensors
    # in the per-expert hot loop.
    for pack in inners:
        pack["_exl3_gate_up_shared_suh"] = bool(
            torch.equal(pack["gate"].suh, pack["up"].suh)
        )
    hidden = int(layer._exl3_hidden_size)
    intermediate = int(layer._exl3_intermediate_local)

    def _ptrs(which: str, attr: str) -> torch.Tensor:
        return torch.tensor(
            [int(getattr(pack[which], attr).data_ptr()) for pack in inners],
            dtype=torch.int64,
            device=device,
        )

    layer._exl3_ptrs = {
        "gate_trellis": _ptrs("gate", "trellis"),
        "gate_suh": _ptrs("gate", "suh"),
        "gate_svh": _ptrs("gate", "svh"),
        "up_trellis": _ptrs("up", "trellis"),
        "up_suh": _ptrs("up", "suh"),
        "up_svh": _ptrs("up", "svh"),
        "down_trellis": _ptrs("down", "trellis"),
        "down_suh": _ptrs("down", "suh"),
        "down_svh": _ptrs("down", "svh"),
    }
    # Short aliases match the native extension terminology and keep the table
    # ABI stable for callers that construct their own RoutedExperts wrapper.
    layer._exl3_ptrs.update(
        {
            "gate_t_ptrs": layer._exl3_ptrs["gate_trellis"],
            "gate_suh_ptrs": layer._exl3_ptrs["gate_suh"],
            "gate_svh_ptrs": layer._exl3_ptrs["gate_svh"],
            "up_t_ptrs": layer._exl3_ptrs["up_trellis"],
            "up_suh_ptrs": layer._exl3_ptrs["up_suh"],
            "up_svh_ptrs": layer._exl3_ptrs["up_svh"],
            "down_t_ptrs": layer._exl3_ptrs["down_trellis"],
            "down_suh_ptrs": layer._exl3_ptrs["down_suh"],
            "down_svh_ptrs": layer._exl3_ptrs["down_svh"],
        }
    )
    idx = int(device.index) if device.index is not None else 0
    if exllamav3_ext is not None and hasattr(
        exllamav3_ext, "exl3_moe_max_concurrency"
    ):
        concurrency = int(exllamav3_ext.exl3_moe_max_concurrency(idx))
        if concurrency < 1:
            concurrency = 1
    else:
        # Native p2b does not consume ExLlamaV3 scratch buffers.
        layer._exl3_fused_temps = None
        layer._exl3_fused_concurrency = 0
        layer._exl3_k = int(layer._exl3_bits)
        return
    key = (str(device), hidden, intermediate, concurrency)
    temps = _FUSED_TEMP_CACHE.get(key)
    if temps is None:
        temps = (
            torch.empty((concurrency, TEMP_ROWS_FUSED, hidden), dtype=torch.float16, device=device),
            torch.empty((concurrency, TEMP_ROWS_FUSED, hidden), dtype=torch.float16, device=device),
            torch.empty((concurrency, TEMP_ROWS_FUSED, intermediate), dtype=torch.float16, device=device),
            torch.empty((concurrency, TEMP_ROWS_FUSED, intermediate), dtype=torch.float16, device=device),
        )
        _FUSED_TEMP_CACHE[key] = temps
    layer._exl3_fused_temps = temps
    layer._exl3_fused_concurrency = concurrency
    if mixed_k:
        # Per-expert K arrays for exl3_moe_mixedk; scalar _exl3_k is invalid.
        layer._exl3_k = None
        layer._exl3_mixedk_unified = True
        k_gate = torch.tensor(
            [int(pack["gate"].trellis.shape[-1]) // 16 for pack in inners],
            dtype=torch.int32,
            device=device,
        )
        k_up = torch.tensor(
            [int(pack["up"].trellis.shape[-1]) // 16 for pack in inners],
            dtype=torch.int32,
            device=device,
        )
        k_down = torch.tensor(
            [int(pack["down"].trellis.shape[-1]) // 16 for pack in inners],
            dtype=torch.int32,
            device=device,
        )
        layer._exl3_K_gate_arr = k_gate
        layer._exl3_K_up_arr = k_up
        layer._exl3_K_down_arr = k_down
    else:
        layer._exl3_k = int(layer._exl3_bits)
        layer._exl3_mixedk_unified = False


def _native_moe_dimensions_supported(
    x2d: torch.Tensor,
    layer: torch.nn.Module,
    inners: list[dict[str, Any]],
    limit: float | None = None,
) -> bool:
    """Supported decode geometry; extension ABI support is checked separately."""
    if x2d.dim() != 2 or not x2d.is_cuda:
        return False
    if limit is not None and (not math.isfinite(limit) or limit < 0):
        return False
    hidden_meta = int(getattr(layer, "_exl3_hidden_size", x2d.shape[1]))
    inter_meta = int(getattr(layer, "_exl3_intermediate_local", 2048))
    rows = int(x2d.shape[0])
    bits = int(getattr(layer, "_exl3_k", getattr(layer, "_exl3_bits", -1)))
    if not (
        rows >= 1
        and int(x2d.shape[1]) == hidden_meta == 4096
        and inter_meta in (1024, 2048)
        and bits in (2, 3, 4)
        and len(inners) > 0
    ):
        return False
    return rows <= _native_moe_max_rows(bits)


# Measured on one GB10 (sm_121) against exllamav3 exl3_moe in the same process, identical
# experts and routing, 288 experts, top-8, hidden 4096, intermediate 2048. Speedup of
# p2b_fused_moe over exl3_moe, median of 50 launches with fresh routing per launch:
#
#   K=2:  m=1 1.69x   m=2 1.21x   m=4 1.23x   m=8 1.06x
#   K=3:  m=1 1.27x   m=2 0.95x   m=4 0.96x   m=8 0.82x
#   K=4:  m=1 1.28x   m=2 1.00x   m=4 1.01x   m=8 0.89x
#
# The native ABI takes one row per launch, so cost grows linearly with rows while exl3_moe
# batches them in a single launch. Those measurements predate the current native kernels,
# so the per-bit cap is opt-in (VLLM_EXL3_NATIVE_MOE_MEASURED_CAP=1) and the default keeps
# the dispatch contract of up to 8 decode rows. Receipt: tools/receipts/ab_moe_gb10.json.
_NATIVE_MOE_MAX_ROWS = {2: 8, 3: 1, 4: 1}
_NATIVE_MOE_CONTRACT_ROWS = 8


def _native_moe_max_rows(bits: int) -> int:
    """Decode row cap for native dispatch: env override, measured cap when opted in, else 8."""
    override = os.environ.get("VLLM_EXL3_NATIVE_MOE_MAX_ROWS")
    if override:
        try:
            value = int(override)
        except ValueError:
            logger.warning(
                "Ignoring non-integer VLLM_EXL3_NATIVE_MOE_MAX_ROWS=%r", override
            )
        else:
            return max(0, value)
    if os.environ.get("VLLM_EXL3_NATIVE_MOE_MEASURED_CAP", "0") == "1":
        return _NATIVE_MOE_MAX_ROWS.get(int(bits), _NATIVE_MOE_CONTRACT_ROWS)
    return _NATIVE_MOE_CONTRACT_ROWS



# MCG is the historical codebook for native packs; the runtime truth is
# ``layer._exl3_codebook_flags``, set once per layer during finalize. Layers that
# predate the flag (or CPU fixtures that build a bare namespace) fall back to MCG,
# matching every other consumer in this module.
_MCG_CODEBOOK_FLAGS = (True, False) * 3


def _apply_native_fused_moe(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    layer: torch.nn.Module,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float | None = None,
) -> torch.Tensor | None:
    """Run the native cooperative kernel for decode rows when it is safe.

    The native ABI consumes one input row and one routing list per launch.  A
    decode batch is therefore submitted as row views into one preallocated
    output tensor.  No per-token pointer/weight tensors are allocated; the only
    conversion is one contiguous int32 routing table for the complete batch.
    Invalid/non-local IDs are clamped to a valid pointer and receive zero
    routing weight, preventing an out-of-bounds read while preserving fallback
    semantics.
    """
    if getattr(layer, "_exl3_mixed_store", None) is not None:
        return None
    if getattr(layer, "_exl3_codebook_flags", _MCG_CODEBOOK_FLAGS) != _MCG_CODEBOOK_FLAGS:
        return None
    module = _load_native_exl3_ext()
    if module is None or not _native_moe_dimensions_supported(
        x2d, layer, inners, limit
    ):
        return None
    intermediate = int(getattr(layer, "_exl3_intermediate_local", 2048))
    clamp_limit = float(limit) if limit is not None else 0.0
    extended_abi = getattr(module, "P2B_MOE_ABI_VERSION", 1) >= 2
    if not extended_abi and (intermediate != 2048 or clamp_limit > 0):
        # A stale .so still accepts the legacy arguments but would interpret TP2
        # pointer tables as 2048-wide weights or silently omit required clipping.
        reason = "local intermediate width/clipping requires native MoE ABI 2; rebuild vllm_exl3_c"
        layer._exl3_native_error = reason
        getattr(logger, "warning_once", logger.warning)(
            "Native EXL3 MoE fallback: %s (intermediate=%s, limit=%s)",
            reason, intermediate, clamp_limit,
        )
        return None
    ptrs = getattr(layer, "_exl3_ptrs", None)
    if not isinstance(ptrs, dict):
        return None
    required = (
        "gate_trellis",
        "gate_suh",
        "gate_svh",
        "up_trellis",
        "up_suh",
        "up_svh",
        "down_trellis",
        "down_suh",
        "down_svh",
    )
    if any(key not in ptrs for key in required):
        return None

    n_exp = len(inners)
    local = map_topk_to_local(ids, n_exp, expert_map).reshape(ids.shape)
    topk = int(local.shape[-1])
    if topk < 1:
        return None
    # p2b_fused_moe reads int32 IDs and fp16 routing weights.  Clamp before
    # conversion so the invalid sentinel cannot wrap into a large int32 value.
    safe_ids = local.clamp(min=0, max=n_exp - 1).to(dtype=torch.int32).contiguous()
    valid = (local >= 0) & (local < n_exp)
    safe_weights = (
        weights.reshape_as(local)
        .to(dtype=torch.float16)
        .mul(valid.to(dtype=torch.float16))
        .contiguous()
    )
    xh = x2d.to(dtype=torch.float16).contiguous()
    native_out = torch.empty_like(xh)
    k = int(getattr(layer, "_exl3_k", getattr(layer, "_exl3_bits", 4)))
    fn = module.p2b_fused_moe
    extra_args = (intermediate, clamp_limit) if extended_abi else ()
    for row in range(int(x2d.shape[0])):
        result = fn(
            xh[row : row + 1],
            native_out[row : row + 1],
            ptrs["gate_trellis"],
            ptrs["gate_suh"],
            ptrs["gate_svh"],
            ptrs["up_trellis"],
            ptrs["up_suh"],
            ptrs["up_svh"],
            ptrs["down_trellis"],
            ptrs["down_suh"],
            ptrs["down_svh"],
            safe_ids[row],
            safe_weights[row],
            k,
            k,
            k,
            True,
            *extra_args,
        )
        # pybind returns the same output tensor, while lightweight test doubles
        # may return a fresh tensor.  Accommodate both without synchronizing.
        if isinstance(result, torch.Tensor) and result is not native_out:
            native_out[row : row + 1].copy_(result.reshape(1, -1))
    return native_out.to(dtype=torch.float32)


_FAT_SCRATCH_CACHE: dict[tuple[str, int, int, int, int], dict[str, torch.Tensor]] = {}


def _fat_scratch(
    device: torch.device, capacity: int, gate: Any
) -> dict[str, torch.Tensor]:
    hidden = int(getattr(gate, "in_features", 4096))
    intermediate = int(getattr(gate, "out_features", 2048))
    k_words = int(gate.trellis.shape[2])
    bucketed_cap = max(256, ((int(capacity) + 255) // 256) * 256)
    key = (str(device), bucketed_cap, intermediate, hidden, k_words)
    scratch = _FAT_SCRATCH_CACHE.get(key)
    if scratch is not None:
        return scratch

    in_tiles, out_tiles, k_words = map(int, gate.trellis.shape)
    scratch = {
        "packed13": torch.empty(
            (in_tiles, 2 * out_tiles, k_words),
            dtype=torch.int16,
            device=device,
        ),
        "svh13": torch.empty(
            2 * intermediate, dtype=torch.float16, device=device
        ),
        "w13": torch.empty(
            (hidden, 2 * intermediate), dtype=torch.float16, device=device
        ),
        "w2": torch.empty(
            (intermediate, hidden), dtype=torch.float16, device=device
        ),
        "h": torch.empty(
            (bucketed_cap, hidden), dtype=torch.float16, device=device
        ),
        "h13": torch.empty(
            (bucketed_cap, hidden), dtype=torch.float16, device=device
        ),
        "gate_up": torch.empty(
            (bucketed_cap, 2 * intermediate), dtype=torch.float32, device=device
        ),
        "act": torch.empty(
            (bucketed_cap, intermediate), dtype=torch.float32, device=device
        ),
        "act_h": torch.empty(
            (bucketed_cap, intermediate), dtype=torch.float16, device=device
        ),
        "h2": torch.empty(
            (bucketed_cap, intermediate), dtype=torch.float16, device=device
        ),
        "down": torch.empty(
            (bucketed_cap, hidden), dtype=torch.float32, device=device
        ),
        "w_gate": torch.empty(
            (hidden, intermediate), dtype=torch.float16, device=device
        ),
        "w_up": torch.empty(
            (hidden, intermediate), dtype=torch.float16, device=device
        ),
        # Contiguous per-projection outputs for the distinct-suh branch: the
        # extension GEMM and Hadamard kernels index row-major contiguous
        # operands, so column slices of ``gate_up`` must not be handed to them.
        "g_tmp": torch.empty(
            (bucketed_cap, intermediate), dtype=torch.float32, device=device
        ),
        "u_tmp": torch.empty(
            (bucketed_cap, intermediate), dtype=torch.float32, device=device
        ),
    }
    _FAT_SCRATCH_CACHE[key] = scratch
    return scratch


def _fat_kernel_available() -> bool:
    native_c = _load_native_exl3_ext()
    if not bool(native_c and hasattr(native_c, "exl3_fat_gemm")):
        return False
    # exl3_fat_gemm requires sm_80+ (ldsm4 + m16n8k16); on Volta the
    # reconstruct+hgemm fallback covers fat experts.
    if os.environ.get("VLLM_EXL3_SM70") == "1":
        return False
    return True


def apply_exl3_batched_fat(
    xh: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    counts_host: list[int],
    inners: list[dict[str, Any]],
    limit: float | None,
    cap: int,
    out: torch.Tensor,
    use_kernel: bool = True,
) -> torch.Tensor:
    """Run fat experts with persistent scratch and accelerated 128x128 CUDA GEMM."""
    native_c = _load_native_exl3_ext()
    ext = load_exllamav3_ext()
    offset = 0
    for e, n_rows in enumerate(counts_host):
        start = offset
        offset += n_rows
        if n_rows <= cap:
            continue

        token_idx = token_sorted[start:offset]
        gate = inners[e]["gate"]
        up = inners[e]["up"]
        down = inners[e]["down"]
        scratch = _fat_scratch(xh.device, n_rows, gate)
        intermediate = int(gate.out_features)

        h = scratch["h"][:n_rows]
        h13 = scratch["h13"][:n_rows]
        torch.index_select(xh, 0, token_idx, out=h)
        shared_suh = inners[e].get("_exl3_gate_up_shared_suh")
        if shared_suh is None:
            # Compatibility fallback for callers that bypass build_exl3_fused_state.
            shared_suh = bool(torch.equal(gate.suh, up.suh))
            inners[e]["_exl3_gate_up_shared_suh"] = shared_suh
        distinct_suh = not shared_suh
        if not distinct_suh:
            ext.had_r_128(h, h13, gate.suh, None, 1.0)

        packed13 = scratch["packed13"]
        out_tiles = int(gate.trellis.shape[1])
        packed13[:, :out_tiles].copy_(gate.trellis)
        packed13[:, out_tiles:].copy_(up.trellis)
        gate_up = scratch["gate_up"][:n_rows]
        svh13 = scratch["svh13"]
        svh13[:intermediate].copy_(gate.svh)
        svh13[intermediate:].copy_(up.svh)

        k = int(getattr(gate, "K", 4))
        mcg = bool(getattr(gate, "mcg", True))
        mul1 = bool(getattr(gate, "mul1", False))

        if (
            not distinct_suh
            and use_kernel
            and k == 4
            and mcg
            and not mul1
            and native_c is not None
            and hasattr(native_c, "exl3_fat_gemm")
        ):
            native_c.exl3_fat_gemm(
                h13, packed13, gate_up, svh13, k, mcg, mul1
            )
        else:
            if distinct_suh:
                gate_h = h13
                up_h = h
                ext.had_r_128(h, gate_h, gate.suh, None, 1.0)
                ext.had_r_128(up_h, up_h, up.suh, None, 1.0)
                w_gate = scratch["w_gate"]
                w_up = scratch["w_up"]
                ext.reconstruct(w_gate, gate.trellis, k, mcg, mul1)
                ext.reconstruct(w_up, up.trellis, k, mcg, mul1)
                # Contiguous temporaries: ext.hgemm / ext.had_r_128 read and
                # write row-major contiguous matrices, and a column slice of
                # ``gate_up`` is neither (see patch_fat_distinct).
                g_tmp = scratch["g_tmp"][:n_rows]
                u_tmp = scratch["u_tmp"][:n_rows]
                ext.hgemm(gate_h, w_gate, g_tmp)
                ext.hgemm(up_h, w_up, u_tmp)
                ext.had_r_128(g_tmp, g_tmp, None, gate.svh, 1.0)
                ext.had_r_128(u_tmp, u_tmp, None, up.svh, 1.0)
            else:
                w13 = scratch["w13"]
                ext.reconstruct(w13, packed13, k, mcg, mul1)
                ext.hgemm(h13, w13, gate_up)
                ext.had_r_128(gate_up, gate_up, None, svh13, 1.0)

        if distinct_suh:
            gate_out = scratch["g_tmp"][:n_rows]
            up_out = scratch["u_tmp"][:n_rows]
        else:
            gate_out = gate_up[:, :intermediate]
            up_out = gate_up[:, intermediate:]
        if limit is not None and limit > 0:
            gate_out.clamp_(max=limit)
            up_out.clamp_(min=-limit, max=limit)
        act = scratch["act"][:n_rows]
        torch.sigmoid(gate_out, out=act)
        act.mul_(gate_out).mul_(up_out)
        act_h = scratch["act_h"][:n_rows]
        act_h.copy_(act)

        h2 = scratch["h2"][:n_rows]
        ext.had_r_128(act_h, h2, down.suh, None, 1.0)
        if (
            not distinct_suh
            and use_kernel
            and k == 4
            and mcg
            and not mul1
            and native_c is not None
            and hasattr(native_c, "exl3_fat_gemm_scatter")
        ):
            native_c.exl3_fat_gemm_scatter(
                h2,
                down.trellis,
                out,
                down.svh,
                token_idx,
                weight_sorted[start:offset],
                k,
                mcg,
                mul1,
            )
        else:
            w2 = scratch["w2"]
            ext.reconstruct(w2, down.trellis, k, mcg, mul1)
            down_out = scratch["down"][:n_rows]
            ext.hgemm(h2, w2, down_out)
            ext.had_r_128(down_out, down_out, None, down.svh, 1.0)
            down_out.mul_(weight_sorted[start:offset].unsqueeze(-1))
            out.index_add_(0, token_idx, down_out)
    return out


def apply_exl3_fused_moe(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    layer: torch.nn.Module,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float | None = None,
) -> torch.Tensor:
    """One exl3_moe launch per layer, with fat-expert fallback. Supports mixed-K via exl3_moe_mixedk."""
    is_mixedk = getattr(layer, "_exl3_mixedk_unified", False)
    if getattr(layer, "_exl3_mixed_store", None) is not None and not is_mixedk:
        raise ValueError("EXL3 mixed-K cannot use the uniform-K fused/fat entry point")
    tokens, hidden = x2d.shape
    n_exp = len(inners)

    # Keep this entry point independently usable by callers that bypass
    # ``apply_exl3_experts`` (for example, custom vLLM runners).
    if get_moe_kernel_backend() == "native":
        try:
            native_out = _apply_native_fused_moe(
                x2d, ids, weights, layer, inners, expert_map, limit
            )
        except Exception as exc:
            native_out = None
            layer._exl3_native_error = repr(exc)
            logger.warning_once(
                "Native EXL3 MoE dispatch failed in fused entry point; "
                "falling back to ExLlamaV3/Python: %s",
                exc,
            )
        if native_out is not None:
            layer._exl3_last_apply = "native"
            return native_out

    try:
        import exllamav3_ext
    except Exception:
        # Direct callers may use this helper without installing ExLlamaV3.
        # Keep the same graceful fallback contract as ``apply_exl3_experts``.
        return apply_exl3_python_loop(
            x2d, ids, weights, inners, expert_map, limit
        )

    ptrs = getattr(layer, "_exl3_ptrs", None)
    temps = getattr(layer, "_exl3_fused_temps", None)
    if not ptrs or temps is None:
        raise RuntimeError("EXL3 fused pointer tables were not built after weight load")

    local = map_topk_to_local(ids, n_exp, expert_map)
    topk = int(ids.shape[-1])
    flat_token = torch.arange(tokens, device=x2d.device, dtype=torch.long).repeat_interleave(topk)
    flat_weight = weights.reshape(-1).to(dtype=torch.float16)
    # scatter_add stays on GPU. torch.bincount can host-stage and break CUDA graphs.
    expert_count = torch.zeros(n_exp + 1, dtype=torch.long, device=local.device)
    expert_count.scatter_add_(
        0, local.long(), torch.ones(local.shape, dtype=torch.long, device=local.device)
    )
    out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x2d.device)
    xh = x2d.contiguous().half()

    counts = expert_count[:n_exp]

    if tokens > TEMP_ROWS_FUSED and bool((counts > TEMP_ROWS_FUSED).any().item()):
        logger.info_once("EXL3 fat-chunk slicing ACTIVE (tokens=%d)" % tokens)
        # Deep-context prefill chunks can route more than TEMP_ROWS_FUSED rows
        # to a single expert. The fused kernel covers at most TEMP_ROWS_FUSED
        # rows per expert, and the old fallback reconstructed whole experts
        # per chunk, stalling prefill by orders of magnitude past ~160k
        # context (the ">163k hang"). Within a slice of <= TEMP_ROWS_FUSED
        # tokens no expert can exceed TEMP_ROWS_FUSED rows (each token adds at
        # most one row per expert), so re-run the fused path per slice.
        # Prefill-only: decode batches are at most the largest capture size,
        # far below TEMP_ROWS_FUSED, and never reach this host sync.
        for s in range(0, tokens, TEMP_ROWS_FUSED):
            e = min(s + TEMP_ROWS_FUSED, tokens)
            out[s:e] = apply_exl3_fused_moe(
                x2d[s:e], ids[s:e], weights[s:e], layer, inners, expert_map, limit
            )
        return out

    fat = counts > FAT_EXPERT_THRESHOLD
    # counts[e] cannot exceed the row count, so with no more rows than the
    # threshold no expert can be fat. Testing that Python-side first keeps the
    # device sync below off the decode path, where graph capture forbids it.
    fat_possible = tokens > FAT_EXPERT_THRESHOLD
    fat_route = torch.zeros_like(local, dtype=torch.bool)
    if fat_possible and bool(fat.any().item()):
        safe_local = local.clamp(min=0, max=max(n_exp - 1, 0))
        fat_route = (local < n_exp) & fat.index_select(0, safe_local)

    # exl3_moe_coop fast path (exllamav3 >= 1.5.0): decode-shaped batches only —
    # the slot scratch is capped at 256, i.e. tokens <= 42 at topk 6, which covers
    # single-stream speculation and light-concurrency traffic. Larger batches and
    # fat routes fall through to the stock path below.
    # Mixed-K layers skip coop and use the mixedk kernel instead.
    if (
        _COOP
        and not is_mixedk
        and tokens * topk <= 256
        and hasattr(exllamav3_ext, "exl3_moe_coop")
        and not (fat_possible and bool(fat.any().item()))
    ):
        flags = getattr(layer, "_exl3_codebook_flags", _MCG_CODEBOOK_FLAGS)
        mcg, mul1 = bool(flags[0]), bool(flags[1])
        inter_dim = int(temps[2].shape[-1])
        if (
            all(bool(flags[i]) == mcg and bool(flags[i + 1]) == mul1 for i in range(0, 6, 2))
            and hidden % 128 == 0
            and inter_dim % 128 == 0
        ):
            k = int(getattr(layer, "_exl3_k", 4))
            slots = tokens * topk
            smax = 4 * slots  # split-k partial rows, per tests/test_moe_coop.py
            dev = x2d.device
            sel_c = local.reshape(tokens, topk).to(torch.int64).contiguous()
            rw_c = flat_weight.reshape(tokens, topk).contiguous()
            had_g = torch.empty((slots, hidden), dtype=torch.float16, device=dev)
            had_u = torch.empty_like(had_g)
            gu_g = torch.empty((smax, 1, inter_dim), dtype=torch.float16, device=dev)
            gu_u = torch.empty_like(gu_g)
            act_out = torch.empty_like(gu_g)
            d_out = torch.empty((smax, 1, hidden), dtype=torch.float32, device=dev)
            ctr = torch.zeros(
                smax * (inter_dim // 128) + tokens * (hidden // 128) + 2 * smax + 3,
                dtype=torch.int32,
                device=dev,
            )
            # Kernel contract: min_expert=-1 disables range filtering and
            # indexes pointer tables by raw sel. Plugin sentinels are
            # n_exp (non-local / EP). Pass [0, n_exp) so those routes
            # contribute zero instead of OOB.
            exllamav3_ext.exl3_moe_coop(
                xh, sel_c, rw_c, 0, int(n_exp), hidden,
                ptrs["gate_trellis"], ptrs["gate_suh"], ptrs["gate_svh"],
                ptrs["up_trellis"], ptrs["up_suh"], ptrs["up_svh"],
                ptrs["down_trellis"], ptrs["down_suh"], ptrs["down_svh"],
                None, None, None,
                k, k, k, mcg, mul1, MOE_ACT_SILU,
                float(limit) if (limit is not None and limit > 0) else 0.0,
                True,
                had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out,
                None, None,
            )
            return out

    # The standard kernel handles non-fat routes. Fat routes are represented by
    # the invalid sentinel with zero weight here and are dispatched exactly once
    # below through the fat GEMM path.
    standard_local = local.masked_fill(fat_route, n_exp)
    standard_weight = flat_weight.masked_fill(fat_route, 0)
    order = standard_local.argsort()
    token_sorted = flat_token[order]
    weight_sorted = standard_weight[order]
    standard_count = torch.zeros(
        n_exp + 1, dtype=torch.long, device=local.device
    )
    standard_count.scatter_add_(
        0,
        standard_local.long(),
        torch.ones(standard_local.shape, dtype=torch.long, device=local.device),
    )
    fn = exllamav3_ext.exl3_moe
    # -1 = unknown active count: max-concurrency grid, no .item() host sync.
    n_active_host = -1 if _exl3_moe_accepts_num_active(fn) else None

    if is_mixedk and hasattr(exllamav3_ext, "exl3_moe_mixedk"):
        # Mixed-K fused dispatch: per-expert K arrays instead of scalar K.
        fn_mk = exllamav3_ext.exl3_moe_mixedk
        args_mk = (
            xh,
            out,
            standard_count,
            token_sorted,
            weight_sorted,
            temps[0],
            temps[1],
            temps[2],
            temps[3],
            MOE_ACT_SILU,
            layer._exl3_K_gate_arr,
            layer._exl3_K_up_arr,
            layer._exl3_K_down_arr,
            ptrs["gate_trellis"],
            ptrs["gate_suh"],
            ptrs["gate_svh"],
            ptrs["up_trellis"],
            ptrs["up_suh"],
            ptrs["up_svh"],
            ptrs["down_trellis"],
            ptrs["down_suh"],
            ptrs["down_svh"],
            *getattr(layer, "_exl3_codebook_flags", _MCG_CODEBOOK_FLAGS),
            float(limit) if (limit is not None and limit > 0) else 0.0,
        )
        tail = _exl3_moe_tail(fn_mk, _exl3_moe_temp_rows(temps))
        n_active_mk = -1 if _exl3_moe_accepts_num_active(fn_mk) else None
        if tail and n_active_mk is None:
            n_active_mk = -1
        if n_active_mk is not None:
            fn_mk(*args_mk, n_active_mk, *tail)
        else:
            fn_mk(*args_mk)
    else:
        k = int(getattr(layer, "_exl3_k", 4))
        args = (
            xh,
            out,
            standard_count,
            token_sorted,
            weight_sorted,
            temps[0],
            temps[1],
            temps[2],
            temps[3],
            MOE_ACT_SILU,
            k,
            k,
            k,
            ptrs["gate_trellis"],
            ptrs["gate_suh"],
            ptrs["gate_svh"],
            ptrs["up_trellis"],
            ptrs["up_suh"],
            ptrs["up_svh"],
            ptrs["down_trellis"],
            ptrs["down_suh"],
            ptrs["down_svh"],
            *getattr(layer, "_exl3_codebook_flags", _MCG_CODEBOOK_FLAGS),
            float(limit) if (limit is not None and limit > 0) else 0.0,
        )
        # exllamav3 >= 1.5.0 takes five more positional arguments after num_active.
        tail = _exl3_moe_tail(fn, _exl3_moe_temp_rows(temps))
        if tail and n_active_host is None:
            n_active_host = -1
        if n_active_host is not None:
            fn(*args, n_active_host, *tail)
        else:
            fn(*args)

    if fat_possible and bool(fat.any().item()):
        fat_order = local.argsort()
        apply_exl3_batched_fat(
            xh,
            flat_token[fat_order],
            flat_weight[fat_order],
            counts.tolist(),
            inners,
            limit,
            FAT_EXPERT_THRESHOLD,
            out,
            use_kernel=_fat_kernel_available(),
        )
    return out


def apply_exl3_experts(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer: torch.nn.Module,
    *,
    limit: float | None = None,
    fused: bool | None = None,
) -> torch.Tensor:
    """Shipped routed-expert apply. `fused=None` honors EXL3_FUSED_MOE."""
    if getattr(layer, "_exl3_mixed_store", None) is not None:
        # Mixed-K with the fused mixedk kernel available — route through the
        # standard fused path which will dispatch to exl3_moe_mixedk.
        if getattr(layer, "_exl3_mixedk_unified", False):
            pass  # fall through to the normal fused/loop dispatch below
        else:
            from .tensor_mixed_k import apply_mixed_reference

            if fused is True:
                raise RuntimeError("EXL3 mixed-K currently supports eager reference execution only")
            return apply_mixed_reference(x, topk_ids, topk_weights, layer, limit=limit)
    elif getattr(layer, "_exl3_mixed_k", False):
        # Standard per-expert-loop mixed-K (not tensor-mixed-K store) — the
        # mixedk kernel path is handled via _exl3_mixedk_unified flag.
        if not getattr(layer, "_exl3_mixedk_unified", False):
            from .tensor_mixed_k import apply_mixed_reference

            if fused is True:
                raise RuntimeError("EXL3 mixed-K currently supports eager reference execution only")
            return apply_mixed_reference(x, topk_ids, topk_weights, layer, limit=limit)
    if _EXL3_PREFILL_SYNC:
        _prefill_sync(int(x.numel() // x.shape[-1]))
    inners = getattr(layer, "_exl3_inners", None)
    if not inners:
        raise RuntimeError("EXL3 experts were not built after weight load")
    tokens, hidden = x.shape[-2], x.shape[-1]
    x2d = x.reshape(tokens, hidden)
    ids = topk_ids.reshape(tokens, -1).to(torch.long)
    weights = topk_weights.reshape(tokens, -1)
    expert_map = pin_exl3_expert_map(layer, x2d.device)

    # Native p2b is a decode-only path.  It is selected explicitly with
    # ``native`` or automatically when the optional extension is installed;
    # unsupported shapes and launch failures fall through to the established
    # ExLlamaV3/Python implementations below.
    backend = get_moe_kernel_backend()
    if backend == "native" and (fused is not False):
        try:
            native_out = _apply_native_fused_moe(
                x2d, ids, weights, layer, inners, expert_map, limit
            )
        except Exception as exc:
            native_out = None
            layer._exl3_native_error = repr(exc)
            getattr(logger, "warning_once", logger.warning)(
                "Native EXL3 MoE dispatch failed; falling back to %s: %s",
                "ExLlamaV3" if _exllamav3_moe_available() else "Python loop",
                exc,
            )
        if native_out is not None:
            layer._exl3_last_apply = "native"
            return native_out.to(dtype=x.dtype)

    have_ptrs = bool(getattr(layer, "_exl3_ptrs", None))
    if fused is True and not have_ptrs:
        raise RuntimeError("EXL3 fused apply requested but pointer tables are missing")
    use_fused = (fused_moe_enabled() if fused is None else bool(fused)) and have_ptrs
    if use_fused:
        try:
            import exllamav3_ext

            use_fused = hasattr(exllamav3_ext, "exl3_moe")
            # exl3_moe is the sm80 block-pipelined kernel; on Volta it
            # would launch with mma/cp.async PTX the device cannot run.
            # Fall through to the per-expert path, which routes through
            # the sm70 GEMV/tiled kernels.
            if use_fused and int(exllamav3_ext.g_get_cc(x.device.index)) < 8:
                use_fused = False
        except Exception:
            use_fused = False
    if use_fused:
        out = apply_exl3_fused_moe(x2d, ids, weights, layer, inners, expert_map, limit)
        layer._exl3_last_apply = "fused"
    else:
        out = apply_exl3_python_loop(x2d, ids, weights, inners, expert_map, limit)
        layer._exl3_last_apply = "loop"
    return out.to(dtype=x.dtype)


def _suffix_from_mapped_name(weight_name: str) -> str:
    tail = weight_name.rsplit(".", 1)[-1]
    for suffix in EXL3_SUFFIXES:
        if tail == suffix or tail.endswith("_" + suffix):
            return suffix
    raise ValueError(f"not an EXL3 packed name: {weight_name}")


def _exl3_pad128(n: int) -> int:
    """EXL3 stores matrices padded to multiples of 128 on both dims."""
    return (int(n) + 127) // 128 * 128


def _prefix_has_suffix(prefix: str, suffix: str) -> bool:
    """Module-path suffix match: "self_attn.o_proj" matches
    "model.layers.3.self_attn.o_proj" but not "...cross_attn.o_proj_x"."""
    return prefix == suffix or prefix.endswith("." + suffix)


@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):
    """Routed-experts-only EXL3/MCG. Dense / shared / attention stay native."""

    def __init__(
        self,
        bits: int = 4,
        codebook: str = "mcg",
        scope: str = "glm53_routed_experts_only",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.bits = int(bits)
        # Attention-family bits (some packs quantize attention output
        # projections at higher precision than the body).
        self.head_bits = int(kwargs.pop("head_bits", 0) or 0)
        # Batched (bmm) layer prefixes → per-rank slice counts; set by
        # the model before layer construction (see create_weights).
        self.bmm_prefixes = dict(kwargs.pop("bmm_prefixes", {}) or {})
        self.codebook = str(codebook)
        self.scope = str(scope)
        # Optional per-layer override, e.g. {"42": 3, "27": 3}. Layers absent
        # from the map use `bits`. This is how a mixed-K checkpoint (K2 base
        # with K3 delta layers) declares itself; the trellis tensors for those
        # layers are shaped for their own K and would fail the load shape
        # check under the base K.
        raw_layer_bits = kwargs.pop("layer_bits", None) or {}
        self.layer_bits: dict[int, int] = {
            int(k): int(v) for k, v in dict(raw_layer_bits).items()
        }
        for layer_idx, layer_k in self.layer_bits.items():
            if layer_k not in (2, 3, 4, 5, 6):
                raise ValueError(
                    f"unsupported EXL3 bits={layer_k} for layer {layer_idx}"
                )
        # Non-routed dense linear config: optional {"modules": [...], "bits": K, "layer_bits": {...},
        # "codebook": "mcg"|"mul1", "layers": {prefix: {"bits": K, "bf16_shards": [...]}, ...}}
        raw_nr_exl3 = kwargs.pop("non_routed_exl3", None) or {}
        self.non_routed_exl3: dict[str, Any] = dict(raw_nr_exl3) if raw_nr_exl3 else {}
        # Validate non-routed bits if present
        nr_bits = self.non_routed_exl3.get("bits")
        if nr_bits is not None and nr_bits not in (2, 3, 4, 5, 6):
            raise ValueError(f"unsupported non_routed_exl3 bits={nr_bits}")
        nr_layer_bits = self.non_routed_exl3.get("layer_bits", {})
        for suffix, k in (nr_layer_bits or {}).items():
            if k not in (2, 3, 4, 5, 6):
                raise ValueError(
                    f"unsupported non_routed_exl3 bits={k} for suffix {suffix}"
                )
        # Validate non-routed layers dict: each value is {"bits": K[, "bf16_shards": [...]]}
        nr_layers = self.non_routed_exl3.get("layers", {})
        for prefix, layer_cfg in (nr_layers or {}).items():
            if not isinstance(layer_cfg, dict):
                raise ValueError(
                    f"non_routed_exl3 layers[{prefix}] must be a dict, got {type(layer_cfg)}"
                )
            layer_bits = layer_cfg.get("bits")
            if layer_bits is not None and layer_bits not in (2, 3, 4, 5, 6):
                raise ValueError(
                    f"unsupported non_routed_exl3 layers[{prefix}] bits={layer_bits}"
                )
        # Validate non-routed codebook
        nr_codebook = self.non_routed_exl3.get("codebook", "mcg")
        if nr_codebook not in ("mcg", "mul1"):
            raise ValueError(
                f"unsupported non_routed_exl3 codebook={nr_codebook!r}; must be 'mcg' or 'mul1'"
            )
        # Optional row-wise trellis embedding table in exllamav3's n-gram format
        # (Qwen3.8-Flash-Next PLE), e.g. {"bits": 5, "num_shards": 128,
        # "rows_per_shard": 2500012, "num_heads": 16, "modules": ["ngram_embedding"]}.
        raw_ngram = kwargs.pop("ngram_embedding", None) or {}
        self.ngram_embedding: dict[str, Any] = dict(raw_ngram) if raw_ngram else {}
        if self.ngram_embedding:
            for key in ("bits", "num_shards", "rows_per_shard", "num_heads"):
                if int(self.ngram_embedding.get(key, 0) or 0) <= 0:
                    raise ValueError(f"ngram_embedding.{key} must be a positive integer")
            if int(self.ngram_embedding["bits"]) not in range(1, 9):
                raise ValueError(
                    f"unsupported ngram_embedding bits={self.ngram_embedding['bits']}"
                )
        self.raw_config = dict(kwargs)
        if self.codebook not in ("mcg", "mul1"):
            raise ValueError(
                f"unsupported codebook={self.codebook!r}; must be 'mcg' or 'mul1'"
            )
        if self.bits not in (2, 3, 4, 5, 6):
            raise ValueError(f"unsupported EXL3 bits={self.bits}")


    def _mtp_expert_method(self, layer, prefix):
        """Quant method for draft/MTP experts, which stay in the base format.

        These are the model's own experts, fp4 for DSV4: packed weights with
        E8M0 block scales, which vLLM loads by looking up w13_weight_scale. The
        non-routed delegate describes fp8 block quantization for the attention
        and dense layers and creates w13_weight_scale_inv instead, so it cannot
        serve these. Prefer MXFP4, whose MoE method creates the expected names,
        and keep the non-routed delegate as the fallback for packs that are not
        fp4. Returns (method, description).
        """
        if str(getattr(self, "mtp_expert_dtype", "fp4")) == "fp4":
            try:
                from vllm.model_executor.layers.quantization.mxfp4 import (
                    Mxfp4Config,
                )

                method = Mxfp4Config().get_quant_method(layer, prefix)
                if method is not None:
                    return method, "mxfp4"
            except Exception as exc:  # pragma: no cover - depends on vLLM build
                if os.environ.get("VLLM_EXL3_LOG_MOE_ROUTING"):
                    print(f"[exl3-routing] mxfp4 delegate unusable: {exc}", flush=True)
        delegate = self._non_routed_delegate()
        if delegate is not None:
            method = delegate.get_quant_method(layer, prefix)
            if method is not None:
                return method, "non_routed"
        return None, "none"

    def get_name(self) -> str:
        return "exl3"

    _LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

    def bits_for_prefix(self, prefix: str) -> int:
        """Per-layer K: `layer_bits` entry for this layer, else the base K."""
        if not self.layer_bits:
            return self.bits
        m = self._LAYER_RE.search(prefix or "")
        if m is None:
            return self.bits
        return self.layer_bits.get(int(m.group(1)), self.bits)

    def _matches_non_routed_exl3(self, prefix: str) -> bool:
        """Check if prefix matches non_routed_exl3: either layers dict keys or modules list."""
        if not self.non_routed_exl3:
            return False
        # Check if prefix is a key in the layers dict
        layers = self.non_routed_exl3.get("layers", {})
        if layers and prefix in layers:
            return True
        # Fall back to suffix matching on modules list
        modules = self.non_routed_exl3.get("modules", [])
        if not modules:
            return False
        return any(_prefix_has_suffix(prefix, m) for m in modules)

    def _bits_for_non_routed(self, prefix: str) -> int:
        """Get K bits for non_routed_exl3 layer, checking layers dict first, then suffix form."""
        if not self.non_routed_exl3:
            return self.bits
        # Check layers dict first
        layers = self.non_routed_exl3.get("layers", {})
        if layers and prefix in layers:
            layer_cfg = layers[prefix]
            if "bits" in layer_cfg:
                return int(layer_cfg["bits"])
            return int(self.non_routed_exl3.get("bits", self.bits))
        # Fall back to suffix matching
        modules = self.non_routed_exl3.get("modules", [])
        matched_suffix = None
        for suffix in modules:
            if _prefix_has_suffix(prefix, suffix):
                matched_suffix = suffix
                break
        if matched_suffix is None:
            return self.bits
        # Check layer_bits override for this suffix
        layer_bits = self.non_routed_exl3.get("layer_bits", {})
        if matched_suffix in layer_bits:
            return int(layer_bits[matched_suffix])
        # Fall back to non_routed_exl3 bits or base bits
        return int(self.non_routed_exl3.get("bits", self.bits))

    def _bf16_shards_for(self, prefix: str) -> list[int]:
        """Get bf16 shard indices for a non_routed_exl3 layer from the layers dict."""
        if not self.non_routed_exl3:
            return []
        layers = self.non_routed_exl3.get("layers", {})
        if layers and prefix in layers:
            layer_cfg = layers[prefix]
            return list(layer_cfg.get("bf16_shards", []))
        return []

    def _resolve_prefix_bits_from_checkpoint(
        self, prefix: str
    ) -> int | list[int] | None:
        """Per-tensor K for one linear, from the checkpoint's trellis width.

        Turboderp calibration packs allocate per-tensor bits (a single
        Qwen3.8-27B pack spans K=4..8 across layers while config declares
        bits=6/head_bits=6). The safetensors header of the prefix's own
        trellis is the ground truth; family heuristics (global bits,
        head_bits) are only fallbacks. Reads headers only — no tensor
        data. Cached per prefix. Returns:
          - int: uniform K for the prefix (single tensor or uniform merged
            group);
          - list[int]: per-shard Ks for a mixed-K merged group, ordered by
            the merged_splits mapping (checkpoint tensor order);
          - None: unresolvable (no trellis found, header unreadable).
        """
        cache = getattr(self, "_prefix_bits_cache", None)
        if cache is None:
            cache = self._prefix_bits_cache = {}
        if prefix in cache:
            return cache[prefix]
        bits = self._resolve_prefix_bits_uncached(prefix)
        cache[prefix] = bits
        return bits

    def _resolve_prefix_bits_uncached(
        self, prefix: str
    ) -> int | list[int] | None:
        try:
            import json
            import re
            import struct

            from vllm.config import get_current_vllm_config

            model_dir = get_current_vllm_config().model_config.model
            index_path = os.path.join(model_dir, "model.safetensors.index.json")
            single_path = os.path.join(model_dir, "model.safetensors")
            if os.path.isfile(index_path):
                with open(index_path) as f:
                    weight_map = json.load(f).get("weight_map", {})
            elif os.path.isfile(single_path):
                weight_map = {}
            else:
                return None

            # Prefix forms: language_model.model.layers.0.mlp.down_proj
            # (vLLM) vs model.language_model.layers.0.mlp.down_proj
            # (checkpoint). Match on the layer index plus the module tail,
            # falling back to a plain suffix match for non-layer prefixes.
            # Merged vLLM linears map to their checkpoint tensor splits
            # (gate_up_proj ships as gate_proj + up_proj, etc.).
            # (name, shard_span): each checkpoint tensor covers `span`
            # consecutive output shards of the merged linear. Spans come
            # from the model's stacked-params mapping (in_proj_qkv covers
            # q,k,v = shards 0-2; in_proj_z covers shard 3; qkv_proj is
            # one shard per q/k/v; gate/up are one shard each).
            merged_splits = {
                "gate_up_proj": (("gate_proj", 1), ("up_proj", 1)),
                "qkv_proj": (("q_proj", 1), ("k_proj", 1), ("v_proj", 1)),
                "in_proj_qkvz": (("in_proj_qkv", 3), ("in_proj_z", 1)),
                "kv_proj": (("k_proj", 1), ("v_proj", 1)),
            }
            m = re.search(r"(?:^|\.)layers\.(\d+)\.(.+)$", prefix or "")
            if m is not None:
                layer_idx, tail = m.group(1), m.group(2)
                parent, _, module = tail.rpartition(".")
                splits = merged_splits.get(module)
                if splits is not None:
                    # Swap the merged basename for its checkpoint splits,
                    # keeping the parent path (mlp., linear_attn., ...).
                    names = tuple(t for t, _ in splits)
                    patterns = tuple(
                        f"layers.{layer_idx}.{parent}.{t}.trellis"
                        for t in names
                    ) if parent else tuple(
                        f"layers.{layer_idx}.{t}.trellis" for t in names
                    )
                else:
                    patterns = (f"layers.{layer_idx}.{tail}.trellis",)
            else:
                # Non-layer prefixes (visual.*, lm_head, ...): anchor the
                # match on the FULL key tail — a bare substring like
                # 'proj.trellis' would cross-match the language model's
                # out_proj/down_proj trellis and false-claim visual
                # linears.
                tail = (prefix or "").rsplit(".", 1)[-1]
                patterns = (f"{tail}.trellis",)

            def matches(key: str) -> bool:
                # mtp.layers.N.* must not match layers.N.*
                if key.startswith("mtp."):
                    return False
                # Dot-anchored: '_qkv.trellis' must not match 'qkv.trellis'
                return any(key.endswith("." + p) or key == p for p in patterns)

            keys = [k for k in weight_map if matches(k)]
            shards: dict[str, list[str]] = {}
            for key, shard in weight_map.items():
                if matches(key):
                    shards.setdefault(shard, []).append(key)
            if not shards and os.path.isfile(single_path):
                shards = {single_path: []}
            resolved_ks: list[int] = []
            for shard, shard_keys in shards.items():
                path = shard if os.path.isabs(shard) else os.path.join(
                    model_dir, shard
                )
                with open(path, "rb") as f:
                    (header_len,) = struct.unpack("<Q", f.read(8))
                    header = json.loads(f.read(header_len))
                if not shard_keys:
                    shard_keys = [
                        k for k in header
                        if k != "__metadata__" and matches(k)
                    ]
                for key in shard_keys:
                    shape = header.get(key, {}).get("shape") or []
                    if len(shape) == 3 and shape[-1] % 16 == 0:
                        # Order by which split pattern the key matched —
                        # weight_map order interleaves q/k/v tensors, but
                        # the caller zips the result with the merged_splits
                        # span order.
                        _pi = next(
                            (
                                i
                                for i, p in enumerate(patterns)
                                if key.endswith("." + p) or key == p
                            ),
                            len(patterns),
                        )
                        resolved_ks.append((shape[-1] // 16, _pi))
            resolved_ks = [
                k for k, _ in sorted(resolved_ks, key=lambda t: t[1])
            ]
            # Merged groups: the caller needs per-shard Ks. Uniform → int;
            # mixed-K merged groups (e.g. q=5/k=7/v=7) → the ordered list,
            # one entry per matched tensor (checkpoint order = shard order
            # for the merged_splits mapping); no trellis at all → None.
            if resolved_ks and len(set(resolved_ks)) == 1:
                return resolved_ks[0]
            if resolved_ks:
                return resolved_ks
            return None
        except Exception as exc:
            import os as _os
            if _os.environ.get("VLLM_EXL3_RESOLVER_DEBUG"):
                import traceback
                print(f"[exl3-resolver] {prefix!r} failed: {exc!r}", flush=True)
                traceback.print_exc()
        return None

    def _resolve_indexer_bits_from_checkpoint(
        self, model_dir: str | None = None
    ) -> int | None:
        """Actual indexer trellis width from the checkpoint header.

        Trellis tile words = 16 * bits, so one safetensors header read
        resolves the family ambiguity: DeepSeek V4 packs the indexer at
        head_bits (80-wide = 5bpw) while Qwen3.8 packs it at the global
        bits (64-wide = 4bpw). Reads headers only — no tensor data.
        Returns the resolved bits, or None when unresolvable.
        """
        try:
            import json
            import struct

            if model_dir is None:
                from vllm.config import get_current_vllm_config

                model_dir = get_current_vllm_config().model_config.model
            index_path = os.path.join(model_dir, "model.safetensors.index.json")
            single_path = os.path.join(model_dir, "model.safetensors")
            if os.path.isfile(index_path):
                with open(index_path) as f:
                    weight_map = json.load(f).get("weight_map", {})
                shards: dict[str, list[str]] = {}
                for key, shard in weight_map.items():
                    if "indexer" in key and key.endswith(".trellis"):
                        shards.setdefault(shard, []).append(key)
            elif os.path.isfile(single_path):
                shards = {single_path: []}
            else:
                return None
            for shard, keys in shards.items():
                path = shard if os.path.isabs(shard) else os.path.join(
                    model_dir, shard
                )
                with open(path, "rb") as f:
                    (header_len,) = struct.unpack("<Q", f.read(8))
                    header = json.loads(f.read(header_len))
                if not keys:
                    keys = [
                        k
                        for k in header
                        if k != "__metadata__"
                        and "indexer" in k
                        and k.endswith(".trellis")
                    ]
                for key in keys:
                    shape = header.get(key, {}).get("shape") or []
                    if len(shape) == 3 and shape[-1] % 16 == 0:
                        return shape[-1] // 16
        except Exception:
            pass
        return None

    def _indexer_head_bits_verified(self) -> bool:
        """True when the pack's indexer family really is packed at head_bits.

        The checkpoint's actual packed width decides; when it cannot be
        read, keep the legacy name-family behavior (apply head_bits).
        """
        resolved = getattr(self, "_indexer_bits_resolved", False)
        if resolved is False:
            resolved = self._resolve_indexer_bits_from_checkpoint()
            self._indexer_bits_resolved = resolved
        if resolved is None:
            return True
        return resolved == self.head_bits

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        # LinearEXL3 uses CUDA >= Ampere; GB10 is SM121.
        # VLLM_EXL3_SM70=1: Volta (sm_70) fork — the sm70 GEMV/tiled
        # kernels in exllamav3-sm70 cover decode; fat GEMM is routed
        # to reconstruct+hgemm on cc < 8.
        import os
        if os.environ.get("VLLM_EXL3_SM70") == "1":
            return 70
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        skip = {
            "bits",
            "head_bits",
            "bmm_prefixes",
            "codebook",
            "scope",
            "quant_method",
            # Some packs ship a large per-tensor ledger here; keep it off the config object.
            "tensor_storage",
            "non_routed_exl3",
            "non_routed_quantization",
            "mtp_experts",
            "mtp_experts_start_layer",
        }
        inst = cls(
            bits=int(config.get("bits", 4)),
            head_bits=int(config.get("head_bits", 0)),
            codebook=str(config.get("codebook", "mcg")),
            scope=str(config.get("scope", "glm53_routed_experts_only")),
            non_routed_exl3=config.get("non_routed_exl3"),
            **{k: v for k, v in config.items() if k not in skip},
        )
        # __init__ swallows unknown kwargs; stash the delegation dict explicitly.
        inst.non_routed_quantization = config.get("non_routed_quantization")
        # Packs that omit the ngram_embedding spec but ledger the table under
        # tensor_storage: derive the spec from the ledger entry so the PLE
        # table (Qwen3.8-Flash-Next) claims through Exl3EmbeddingMethod.
        if not inst.ngram_embedding:
            inst.ngram_embedding = _derive_ngram_embedding_spec(
                config.get("tensor_storage") or {}
            ) or {}
        # "bf16_as_stored": dense linears are BF16 tensors; never delegate them
        # (the delegate still serves source-format MTP experts).
        inst.non_routed_dtype_policy = str(config.get("non_routed_dtype_policy", ""))
        # Mixed-format packs: draft/MTP blocks appended past the main stack can
        # keep their experts in the source format (e.g. MXFP4). Declare
        # mtp_experts: "source" plus mtp_experts_start_layer: <first draft
        # layer index>; those layers delegate to non_routed_quantization.
        inst.mtp_experts = str(config.get("mtp_experts", "exl3"))
        inst.mtp_experts_start_layer = config.get("mtp_experts_start_layer")
        return inst

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        method = str((hf_quant_cfg or {}).get("quant_method", "")).lower()
        if method == "exl3":
            return "exl3"
        return None

    def _ngram_embedding_spec(self, prefix: str) -> dict[str, Any] | None:
        """The n-gram table spec if ``prefix`` names one of its modules, else None."""
        spec = getattr(self, "ngram_embedding", None) or {}
        if not spec:
            return None
        modules = list(spec.get("modules") or ["ngram_embedding"])
        for m in modules:
            if prefix == m or prefix.endswith("." + m):
                return spec
        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        try:
            from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        except ImportError:
            RoutedExperts = ()  # fork without the RoutedExperts layer (dense-only)
        from vllm.model_executor.layers.fused_moe import FusedMoE

        if isinstance(layer, (RoutedExperts, FusedMoE)):
            # Draft/MTP blocks construct with plain layers.N prefixes (the
            # mtp_block name appears only in parameter paths), so gate by
            # declared layer index, never by name.
            if getattr(self, "mtp_experts", "exl3") == "source":
                _start = getattr(self, "mtp_experts_start_layer", None)
                _lm = re.search(r"layers\.(\d+)\.", prefix)
                _by_index = bool(
                    _start is not None and _lm and int(_lm.group(1)) >= int(_start)
                )
                if _by_index:
                    dm, how = self._mtp_expert_method(layer, prefix)
                    if dm is not None:
                        if os.environ.get("VLLM_EXL3_LOG_MOE_ROUTING"):
                            print(
                                f"[exl3-routing] MoE {prefix} -> source via {how} "
                                f"(by_index={_by_index})",
                                flush=True,
                            )
                        return dm
            if os.environ.get("VLLM_EXL3_LOG_MOE_ROUTING"):
                print(f"[exl3-routing] MoE {prefix} -> exl3", flush=True)
            return Exl3MoEMethod(
                layer.moe_config, self, bits=self.bits_for_prefix(prefix)
            )
        # Quantized LM head: the checkpoint carries head.trellis etc.
        # (VocabParallelEmbedding isn't LinearBase, so it needs its own
        # branch). The pack quantizes it at head_bits.
        _lp = getattr(layer, "prefix", "") or ""
        if _lp.endswith("lm_head"):
            _hb = getattr(self, "head_bits", 0) or self.bits
            return Exl3LinearMethod(self, bits=_hb)
        if isinstance(layer, LinearBase):
            # Check if this LinearBase should use non_routed_exl3
            if self._matches_non_routed_exl3(prefix):
                bits = self._bits_for_non_routed(prefix)
                layer._exl3_prefix = prefix
                return Exl3LinearMethod(self, bits=bits)
            # Packs quantized end-to-end by exllamav3 carry no
            # non_routed_exl3 spec: every dense linear has trellis
            # tensors in the checkpoint. Default those to exl3 with
            # the global bits; packs that keep dense layers native
            # set non_routed_exl3.exclude or ship no trellis for them.
            if not self.non_routed_exl3 and getattr(
                    self, "non_routed_dtype_policy", "") != "bf16_as_stored":
                layer._exl3_prefix = prefix
                # Packs distinguish attention-family bits (head_bits)
                # from the global bits; attention/compressor linears
                # take head_bits when declared (verified against the
                # checkpoint: wq_a/wkv trellis k_words=80 = 5bpw).
                bits = self.bits
                hb = getattr(self, "head_bits", 0)
                if hb and "indexer." in prefix:
                    # The indexer's packing is ambiguous across packs:
                    # DeepSeek V4 packs it at head_bits (80-wide = 5bpw),
                    # Qwen3.8 at the global bits (64-wide = 4bpw). Decide
                    # from the checkpoint's actual trellis width — note
                    # "_attn." would otherwise swallow self_attn.indexer.*
                    # into the attention family before this clause.
                    if self._indexer_head_bits_verified():
                        bits = hb
                elif hb and ("/attn." in prefix or ".attn." in prefix
                             or "_attn." in prefix
                             or "compressor." in prefix
                             or "shared_expert" in prefix
                             # Vision towers ship whole-tower K=head_bits
                             # (attn, mlp, and merger alike; verified: every
                             # visual trellis is 96-wide in the k6 pack).
                             or "visual." in prefix
                             ):
                    bits = hb
                # Per-tensor calibration packs (per-layer K spanning the
                # global/head_bits pair) carry the real K in each trellis
                # width; the checkpoint wins over every family heuristic.
                # Mixed-K merged groups resolve to a per-tensor K LIST —
                # consumed by create_weights (which re-queries the resolver
                # and expands via the shard-span table); the method-level
                # bits stays scalar.
                resolved = self._resolve_prefix_bits_from_checkpoint(prefix)
                if resolved is None:
                    # No trellis for this prefix in the checkpoint: the
                    # pack keeps this linear native (e.g. Qwen3-VL towers
                    # ship raw weights). Unclaim — claiming would pad the
                    # raw weight into EXL3 staging geometry and fail.
                    import os as _os
                    if _os.environ.get("VLLM_EXL3_RESOLVER_DEBUG"):
                        print(f"[exl3-unclaim] {prefix}", flush=True)
                    return UnquantizedLinearMethod()
                if isinstance(resolved, int):
                    bits = resolved
                return Exl3LinearMethod(self, bits=bits)
            if getattr(self, "non_routed_dtype_policy", "") == "bf16_as_stored":
                import os as _os
                if _os.environ.get("VLLM_EXL3_RESOLVER_DEBUG"):
                    print(f"[exl3-unquant] bf16_as_stored {prefix}", flush=True)
                return UnquantizedLinearMethod()
            d = self._non_routed_delegate()
            if d is not None:
                m = d.get_quant_method(layer, prefix)
                if m is not None:
                    return m
            import os as _os
            if _os.environ.get("VLLM_EXL3_RESOLVER_DEBUG"):
                print(
                    f"[exl3-unquant] delegate-miss {prefix} "
                    f"nr_exl3={self.non_routed_exl3!r} "
                    f"policy={getattr(self, 'non_routed_dtype_policy', '')!r} "
                    f"matches_nr={self._matches_non_routed_exl3(prefix)}",
                    flush=True,
                )
            return UnquantizedLinearMethod()
        # Embedding-family layers. vLLM only consults quant_config for these when
        # the model passes it (qwen4_exp needs quant_config= on ParallelLMHead and
        # on the PLE table). lm_head is an ordinary trellis linear; the PLE n-gram
        # table is the row-wise exllamav3 format served by Exl3EmbeddingMethod.
        try:
            from vllm.model_executor.layers.vocab_parallel_embedding import (
                ParallelLMHead,
                VocabParallelEmbedding,
            )
        except ImportError:  # pragma: no cover
            return None
        if isinstance(layer, ParallelLMHead):
            if self._matches_non_routed_exl3(prefix):
                layer._exl3_prefix = prefix
                return Exl3LinearMethod(self, bits=self._bits_for_non_routed(prefix))
            # Packs quantized end-to-end ship the head tensors
            # (lm_head.trellis/suh/svh/mul1) without a non_routed_exl3
            # spec — claim those too, at head_bits.
            _hb = getattr(self, "head_bits", 0) or self.bits
            layer._exl3_prefix = prefix
            return Exl3LinearMethod(self, bits=_hb)
        if isinstance(layer, VocabParallelEmbedding):
            spec = self._ngram_embedding_spec(prefix)
            if spec is not None:
                layer._exl3_prefix = prefix
                return Exl3EmbeddingMethod(self, spec)
            return None
        return None

    def _non_routed_delegate(self):
        # Packs that keep non-routed weights in the official source format
        # (e.g. DeepSeek block-FP8) declare it under
        # ``quantization_config.non_routed_quantization``; delegate those
        # layers to the matching quant method so arch-specific fp8 forward
        # paths get real scale tensors. Absent key = unquantized (GLM).
        if not hasattr(self, "_nr_delegate_cached"):
            self._nr_delegate_cached = None
            nrq = getattr(self, "non_routed_quantization", None)
            if isinstance(nrq, dict) and nrq.get("quant_method"):
                from vllm.model_executor.layers.quantization import (
                    get_quantization_config,
                )
                name = str(nrq["quant_method"])
                try:
                    cls = get_quantization_config(name)
                except Exception as exc:
                    raise RuntimeError(
                        "Unable to load declared non_routed_quantization delegate "
                        f"quant_method={name!r} config={nrq!r}"
                    ) from exc
                if cls is None:
                    raise ValueError(
                        "Declared non_routed_quantization delegate is unavailable: "
                        f"quant_method={name!r} config={nrq!r}"
                    )
                try:
                    self._nr_delegate_cached = cls.from_config(dict(nrq))
                except Exception as exc:
                    raise ValueError(
                        "Invalid declared non_routed_quantization delegate "
                        f"quant_method={name!r} config={nrq!r}"
                    ) from exc
        return self._nr_delegate_cached


# Mirrors the checkpoint-name resolution of vLLM's RoutedExperts.load_weights
# (vllm-project/vllm, Apache-2.0); see THIRD_PARTY_NOTICES.md.
def _exl3_routed_experts_loader(layer: torch.nn.Module):
    """Per-expert ``load_weights`` for a RoutedExperts layer holding EXL3 tensors.

    Mirrors vLLM's ``RoutedExperts.load_weights`` name resolution but never takes its
    fused (3-D) branch: an EXL3 checkpoint always stores one tensor per expert.
    """

    def load_weights(weights):
        try:
            mapping = layer.get_expert_mapping(include_fused=True)
        except TypeError:
            mapping = layer.get_expert_mapping()
        except AttributeError:
            # Fork builds: FusedMoE carries the mapping as an attribute
            # (self.expert_mapping) instead of a get_expert_mapping method.
            mapping = layer.expert_mapping
        layer_name = str(getattr(layer, "layer_name", ""))
        for expert_name, loaded_weight in weights:
            qual_name = f"{layer_name}.{expert_name}" if layer_name else expert_name
            for param_name, weight_name, expert_id, shard_id in mapping:
                if weight_name not in qual_name:
                    continue
                full_name = qual_name.replace(weight_name, param_name)
                local_name = full_name.removeprefix(f"{layer_name}.")
                param = getattr(layer, local_name, None)
                if param is None:
                    if local_name.endswith(("w13_bias", "w2_bias")):
                        break
                    raise AttributeError(
                        f"EXL3 routed experts {layer_name!r} has no parameter "
                        f"{local_name!r} for checkpoint weight {qual_name!r}"
                    )
                ok = param.weight_loader(
                    param=param,
                    loaded_weight=loaded_weight,
                    weight_name=full_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if ok:
                    yield local_name
                break

    return load_weights




def _marker_row_host(marker: torch.Tensor) -> torch.Tensor:
    """_BATCHED_MARKERS: one host transfer for the whole marker block."""
    return marker.detach().reshape(-1).to("cpu", non_blocking=False)


def _marker_or_none_host(host_row: torch.Tensor, idx: int):
    v = int(host_row[idx].item()) if 0 <= idx < host_row.numel() else 0
    return v if v != 0 else None


def _marker_tensor_or_none_host(host_row: torch.Tensor, idx: int, device):
    v = int(host_row[idx].item()) if 0 <= idx < host_row.numel() else 0
    if v == 0:
        return None
    return torch.tensor([v], dtype=torch.int32, device=device)


def _moe_marker_or_none(marker: torch.Tensor):
    """A codebook marker tensor if it was loaded (non-zero), else None."""
    return marker if int(marker.reshape(-1)[0].item()) != 0 else None


def _check_moe_codebook_markers(mcg: torch.Tensor, mul1: torch.Tensor, what: str) -> None:
    """Every expert tensor carries exactly one codebook marker with the known value."""
    mcg_v = mcg.reshape(-1)
    mul1_v = mul1.reshape(-1)
    mcg_set = mcg_v != 0
    mul1_set = mul1_v != 0
    if bool((mcg_set & mul1_set).any()):
        raise RuntimeError(f"EXL3 {what}: an expert tensor has both mcg and mul1 markers")
    if bool((~mcg_set & ~mul1_set).any()):
        raise RuntimeError(
            f"EXL3 {what}: an expert tensor has no codebook marker (mcg or mul1 never loaded)"
        )
    if bool((mcg_v[mcg_set] != MCG_MARKER_SIGNED_INT32).any()):
        raise RuntimeError(
            f"EXL3 {what}: mcg marker is not the MCG int32 {MCG_MARKER_SIGNED_INT32}; "
            "packed ABI mismatch"
        )
    if bool((mul1_v[mul1_set] != MUL1_MARKER_SIGNED_INT32).any()):
        raise RuntimeError(
            f"EXL3 {what}: mul1 marker is not the mul1 int32 {MUL1_MARKER_SIGNED_INT32}; "
            "packed ABI mismatch"
        )


class Exl3MoEMethod(FusedMoEMethodBase):
    """Packed MCG trellis experts: create/load packed tensors, LinearEXL3 apply."""

    def __init__(
        self, moe, quant_config: Exl3Config, bits: int | None = None
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        # One method instance per RoutedExperts layer, so this is per-layer K.
        self.bits = int(bits) if bits is not None else quant_config.bits
        self._logged = False

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config,
    ) -> tuple[int, int]:
        # ALIGNED (fork scheme, run-110-validated): round the per-rank
        # intermediate up to the 128-block boundary so the packed trellis
        # tiles stay kernel-legal; the loader's block-aligned copy-TRUE
        # branch fills the aligned buffers and the act-mask owns the real
        # channel window. Identity here desynchronized the loader's
        # block-aligned slices from the buffer geometry.
        aligned = -(-intermediate_size_per_partition // 128) * 128
        return hidden_size, aligned

    def get_fused_moe_quant_config(self, layer: "RoutedExperts") -> FusedMoEQuantConfig | None:
        return None

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype
        from .tensor_mixed_k import create_mixed_weights, tensor_mixed_k_enabled

        # Hadamard-aligned uneven TP (VLLM_EXL3_MOE_TP_ALIGN): vLLM hands every
        # rank the equal chunk; replace it with this rank's block-aligned width.
        align = _moe_tp_align()
        if align > 0 and not getattr(layer, "use_ep", False):
            tp_rank, tp_size = _resolve_tp_geometry(layer)
            full = intermediate_size_per_partition * tp_size
            if tp_size > 1 and full % align == 0:
                layer._exl3_tp_rotation = moe_tp_rotation(layer, tp_size)
                tp_rank = (tp_rank + layer._exl3_tp_rotation) % tp_size
                offset, local = aligned_tp_split(full, tp_rank, tp_size, align)
                logger.info(
                    "EXL3 aligned MoE TP: rank %d/%d intermediate %d -> %d (offset %d)",
                    tp_rank, tp_size, intermediate_size_per_partition, local, offset,
                )
                intermediate_size_per_partition = local

        if tensor_mixed_k_enabled():
            # Only the opt-in exact-width store needs 128-aligned local dims; the
            # default uniform-K path keeps main's 16-alignment contract untouched.
            from .deepseek_v41 import exllamav3_fused_geometry_supported

            if not exllamav3_fused_geometry_supported(
                hidden_size, intermediate_size_per_partition
            ):
                raise ValueError(
                    "EXL3 routed transforms require 128-aligned local dimensions; "
                    f"hidden={hidden_size} intermediate_local={intermediate_size_per_partition}. "
                    "Use whole-expert EP or an aligned TP partition; no padding/truncation is safe."
                )
            create_mixed_weights(
                self, layer, num_experts, hidden_size,
                intermediate_size_per_partition, extra_weight_attrs,
            )
            return
        if hidden_size % 16 or intermediate_size_per_partition % 16:
            raise ValueError(
                "EXL3 trellis tiles are 16-wide; "
                f"hidden={hidden_size} intermediate_local={intermediate_size_per_partition}"
            )
        # Default/base K from config. Real DSV4.1 4.75bpw packs are mixed-K
        # even within a single expert (w1/w2/w3 can differ). Trellis storage is
        # therefore ragged and sized on load from the checkpoint tensor itself.
        k_words = self.bits * 16
        in_tiles = hidden_size // 16
        out_tiles = intermediate_size_per_partition // 16

        extra = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}

        # Suh/svh/markers stay stacked (shape independent of packed K).
        # Channel buffers (svh/suh along the intermediate dim) are padded to
        # the 128-block boundary: the loader's block-aligned copy slices
        # [lo:hi] with hi rounded up to 128, so the buffers must hold the
        # aligned width (zero tail). The trellis stays ragged — sized from
        # the checkpoint on load.
        _chan_aligned = -(-intermediate_size_per_partition // 128) * 128
        w13_suh = Parameter(
            torch.empty(num_experts, 2, hidden_size, dtype=torch.float16),
            requires_grad=False,
        )
        w13_svh = Parameter(
            torch.zeros(
                num_experts, 2, _chan_aligned, dtype=torch.float16
            ),
            requires_grad=False,
        )
        w13_mcg = Parameter(
            torch.zeros(num_experts, 2, 1, dtype=torch.int32),
            requires_grad=False,
        )
        w13_mul1 = Parameter(
            torch.zeros(num_experts, 2, 1, dtype=torch.int32),
            requires_grad=False,
        )
        w2_suh = Parameter(
            torch.zeros(
                num_experts, _chan_aligned, dtype=torch.float16
            ),
            requires_grad=False,
        )
        w2_svh = Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.float16),
            requires_grad=False,
        )
        w2_mcg = Parameter(
            torch.zeros(num_experts, 1, dtype=torch.int32),
            requires_grad=False,
        )
        w2_mul1 = Parameter(
            torch.zeros(num_experts, 1, dtype=torch.int32),
            requires_grad=False,
        )

        # Dummy named parameters so expert_params_mapping still resolves
        # ``w13_trellis`` / ``w2_trellis``. Real trellis payloads live in the
        # ragged ParameterLists below and are replaced with exact shapes on load.
        w13_trellis = Parameter(torch.empty(0, dtype=torch.int16), requires_grad=False)
        w2_trellis = Parameter(torch.empty(0, dtype=torch.int16), requires_grad=False)

        gate_trellis = torch.nn.ParameterList(
            [
                Parameter(torch.empty(0, dtype=torch.int16), requires_grad=False)
                for _ in range(num_experts)
            ]
        )
        up_trellis = torch.nn.ParameterList(
            [
                Parameter(torch.empty(0, dtype=torch.int16), requires_grad=False)
                for _ in range(num_experts)
            ]
        )
        down_trellis = torch.nn.ParameterList(
            [
                Parameter(torch.empty(0, dtype=torch.int16), requires_grad=False)
                for _ in range(num_experts)
            ]
        )
        layer.gate_trellis = gate_trellis
        layer.up_trellis = up_trellis
        layer.down_trellis = down_trellis
        # Staging for arena pack: exact per-expert tensors held briefly on host,
        # then copied into contiguous per-shape arenas in process_weights.
        layer._exl3_trellis_staging = {"gate": {}, "up": {}, "down": {}}
        layer._exl3_gate_trellis_arenas = []
        layer._exl3_up_trellis_arenas = []
        layer._exl3_down_trellis_arenas = []
        layer._exl3_trellis_arena_stats = {}
        layer._exl3_trellis_alloc_count_before = 0

        packed = {
            "w13_trellis": w13_trellis,
            "w13_suh": w13_suh,
            "w13_svh": w13_svh,
            "w13_mcg": w13_mcg,
            "w13_mul1": w13_mul1,
            "w2_trellis": w2_trellis,
            "w2_suh": w2_suh,
            "w2_svh": w2_svh,
            "w2_mcg": w2_mcg,
            "w2_mul1": w2_mul1,
        }
        for name, param in packed.items():
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra)
            param.weight_loader = self._load_exl3
            param._exl3_owner = layer
        for plist in (gate_trellis, up_trellis, down_trellis):
            for param in plist:
                set_weight_attrs(param, extra)
                param.weight_loader = self._load_exl3
                param._exl3_owner = layer
        if hasattr(layer, "w13_weight") or hasattr(layer, "w2_weight"):
            raise RuntimeError("EXL3 create_weights must not allocate dense expert weights")

        layer._exl3_hidden_size = hidden_size
        layer._exl3_intermediate_local = intermediate_size_per_partition
        layer._exl3_in_tiles = in_tiles
        layer._exl3_out_tiles = out_tiles
        layer._exl3_k_words = k_words  # config default only; actual K is per-trellis
        layer._exl3_bits = self.bits
        layer._exl3_mixed_k = False
        layer._exl3_n_experts = int(num_experts)
        # Linear EP placement offset for checkpoint prescan (local->global).
        if not hasattr(layer, "starting_expert_offset"):
            try:
                from vllm.distributed.parallel_state import get_ep_group

                ep = get_ep_group()
                layer.starting_expert_offset = int(ep.rank) * int(num_experts)
            except Exception:
                layer.starting_expert_offset = 0
        # Header-only shape scan (no allocation yet). Arenas are created on the
        # first trellis load after the module has been moved to its exec device,
        # so a later layer.to(device) cannot clone views apart.
        layer._exl3_trellis_shapes_pending = None
        if (
            _exl3_trellis_arena_enabled()
            and os.environ.get("VLLM_EXL3_ARENA_PRESCAN", "1") != "0"
        ):
            shapes = _try_prescan_trellis_shapes(layer, int(num_experts))
            if shapes is not None:
                layer._exl3_trellis_shapes_pending = shapes
                logger.info(
                    "EXL3 trellis arena PRESCAN shapes ready for %s experts "
                    "(alloc deferred until first load)",
                    num_experts,
                )
        from .tensor_metadata import current_tensor_metadata_provider
        if current_tensor_metadata_provider() is not None:
            if _exl3_trellis_arena_enabled() and layer._exl3_trellis_shapes_pending is not None:
                layer._exl3_require_direct_fill = True
            else:
                logger.warning(
                    "EXL3 constructor plan missing for %s; continuing without direct-fill (draft/MTP)",
                    getattr(layer, "prefix", None) or getattr(layer, "layer_name", "?"),
                )
        # vLLM's generic RoutedExperts.load_weights treats any 3-D checkpoint
        # tensor as fused stacked experts and unbinds it per expert; an EXL3
        # per-expert trellis is 3-D by construction. Route this layer's tensors
        # through a per-expert loader instead (instance attribute shadows the
        # class method for AutoWeightsLoader; direct-calling models are unaffected).
        layer.load_weights = _exl3_routed_experts_loader(layer)

    def _load_exl3(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str = "w1",
        expert_id: int = 0,
        return_success: bool = False,
    ) -> bool | None:
        layer = param
        # param is the Parameter; expert_id is already physical. Map to local
        # via the owning module if present on the weight_loader closure... we
        # look up from param's __dict__ after register. RoutedExperts.weight_loader
        # maps global→local; glm5next calls *our* loader, so map here.
        owner = getattr(param, "_exl3_owner", None)
        if owner is not None:
            local_id = owner._map_global_expert_id_to_local_expert_id(expert_id)
            if local_id == -1:
                return False if return_success else None
            expert_id = local_id

        owner_mod = owner if owner is not None else getattr(param, "_exl3_owner", None)
        tp_rank, tp_size = _resolve_tp_geometry(owner_mod, param)
        if tp_size > 1 and getattr(owner_mod, "_exl3_tp_rotation", 0):
            # Same per-layer chunk rotation as create_weights (memory balance).
            tp_rank = (tp_rank + owner_mod._exl3_tp_rotation) % tp_size
        if getattr(owner_mod, "use_ep", False):
            # Expert parallel: experts are whole; feature slicing must not run
            # (shard_exl3_col/row would quarter already-whole expert tensors).
            tp_rank, tp_size = 0, 1
        suffix = _suffix_from_mapped_name(weight_name)
        store = getattr(owner, "_exl3_mixed_store", None)
        if store is not None:
            store.check_expert_map(getattr(owner, "expert_map", None))
            store.load(expert_id, shard_id, suffix, loaded_weight, param.device)
            return True if return_success else None
        # Avoid an early full-tensor .contiguous() copy. On GB10 UMA that
        # transient host copy sits beside the eventual device payload and was
        # observed to push MemAvailable under the 16 GiB abort cliff.
        loaded = loaded_weight.detach()
        if suffix in ("mcg", "mul1"):
            # Codebook markers are scalars ([] or [1]); keep the value per expert
            # tensor so process_weights_after_loading can pick the codebook.
            if owner_mod is None:
                raise RuntimeError("EXL3 marker load missing owner module")
            if shard_id in ("w1", "w3"):
                dest = getattr(owner_mod, "w13_" + suffix).data[
                    expert_id, 0 if shard_id == "w1" else 1
                ]
            elif shard_id == "w2":
                dest = getattr(owner_mod, "w2_" + suffix).data[expert_id]
            else:
                raise ValueError(f"unknown EXL3 shard_id={shard_id}")
            dest.fill_(int(loaded.reshape(-1)[0].item()) if loaded.numel() else 0)
            return True if return_success else None

        if suffix == "trellis":
            # Exact checkpoint shape per expert. With arenas enabled, stage on
            # host and pack into contiguous per-shape arenas later (views keep
            # heterogeneous K / expert IDs). Legacy path allocates one Parameter
            # per expert immediately.
            if owner_mod is None:
                raise RuntimeError("EXL3 trellis load missing owner module")
            if _exl3_mem_waterfall_enabled():
                _exl3_mem_snapshot("BEFORE_SOURCE", owner_mod)
            if shard_id in ("w1", "w3"):
                sharded = shard_exl3_col(loaded, suffix, tp_rank, tp_size, aligned=True)
                plist = owner_mod.gate_trellis if shard_id == "w1" else owner_mod.up_trellis
            elif shard_id == "w2":
                sharded = shard_exl3_row(loaded, suffix, tp_rank, tp_size, aligned=True)
                plist = owner_mod.down_trellis
            else:
                raise ValueError(f"unknown EXL3 shard_id={shard_id}")
            if _exl3_mem_waterfall_enabled():
                _exl3_mem_snapshot("AFTER_SOURCE_OPEN", owner_mod)
            # Arena/direct-fill staging and the tile validation belong to
            # the arena/mixed-K feature path. The legacy fork path loads
            # via the block-aligned copy-TRUE branch below, whose geometry
            # (aligned buffers, act-mask) differs from the ragged contract
            # these checks enforce — running them on the legacy path
            # rejected valid shards or staged every expert on host.
            use_arena = _exl3_trellis_arena_enabled()
            if use_arena:
                in_tiles = int(getattr(owner_mod, "_exl3_in_tiles", 0))
                out_tiles = int(getattr(owner_mod, "_exl3_out_tiles", 0))
                if shard_id in ("w1", "w3"):
                    expect_prefix = (in_tiles, out_tiles)
                else:
                    expect_prefix = (out_tiles, in_tiles)
                if tuple(sharded.shape[:2]) != expect_prefix:
                    raise RuntimeError(
                        f"EXL3 trellis tile mismatch {weight_name} shard={shard_id} "
                        f"expert={expert_id}: got {tuple(sharded.shape)} "
                        f"expected prefix {expect_prefix}+K_words"
                    )
                if int(sharded.shape[-1]) % 16 != 0:
                    raise RuntimeError(
                        f"EXL3 trellis K_words not multiple of 16: {tuple(sharded.shape)}"
                    )
            if getattr(owner_mod, "_exl3_require_direct_fill", False) and not use_arena:
                raise RuntimeError("attested EXL3 direct-fill cannot disable final arenas")
            if use_arena:
                proj = _proj_from_shard_id(shard_id)
                owner_mod._exl3_trellis_alloc_count_before = int(
                    getattr(owner_mod, "_exl3_trellis_alloc_count_before", 0)
                ) + 1
                # Materialize deferred prescan plan on the exec device once.
                pending = getattr(owner_mod, "_exl3_trellis_shapes_pending", None)
                if (
                    pending is not None
                    and getattr(owner_mod, "_exl3_trellis_arena_plan", None) is None
                ):
                    prepare_trellis_arena_plan(owner_mod, pending)
                    owner_mod._exl3_trellis_shapes_pending = None
                    logger.info(
                        "EXL3 trellis arenas allocated on %s: arenas=%s final_bytes=%s",
                        owner_mod.w13_suh.device,
                        owner_mod._exl3_trellis_arena_stats.get("allocations_after"),
                        owner_mod._exl3_trellis_arena_stats.get("final_bytes"),
                    )
                # Preferred path: plan exists -> copy straight into FINAL slot.
                if getattr(owner_mod, "_exl3_trellis_arena_plan", None) is not None:
                    if _exl3_mem_waterfall_enabled():
                        _exl3_mem_snapshot("AFTER_DEST_ALLOC", owner_mod)
                        _exl3_mem_snapshot("AFTER_READ", owner_mod)
                        _exl3_mem_snapshot("AFTER_LAYOUT_CONVERSION", owner_mod)
                    _direct_fill_trellis_slot(owner_mod, proj, int(expert_id), sharded)
                    if _exl3_mem_waterfall_enabled():
                        _exl3_mem_snapshot("AFTER_COPY_TO_FINAL", owner_mod)
                    del loaded, sharded, loaded_weight
                    if _exl3_mem_waterfall_enabled():
                        _exl3_mem_snapshot("AFTER_TEMP_DELETE", owner_mod)
                        _exl3_mem_snapshot("AFTER_GC", owner_mod)
                    return True if return_success else None

                if getattr(owner_mod, "_exl3_require_direct_fill", False):
                    raise RuntimeError("attested EXL3 direct-fill plan missing; CPU staging is forbidden")
                # Fallback: stage on host; pack in process_weights_after_loading.
                _fb = int(sharded.numel()) * int(sharded.element_size())
                _DIRECT_FILL_STATS["DIRECT_FILL_FALLBACK_CALLS"] += 1
                _DIRECT_FILL_STATS["DIRECT_FILL_FALLBACK_BYTES"] += _fb
                logger.warning(
                    "EXL3 trellis staging fallback (no arena plan) layer=%s "
                    "proj=%s expert=%s — host Anon coexistence risk on UMA",
                    getattr(owner_mod, "layer_name", None)
                    or getattr(owner_mod, "prefix", "?"),
                    proj,
                    expert_id,
                )
                if _exl3_mem_waterfall_enabled():
                    _exl3_mem_snapshot("AFTER_DEST_ALLOC", owner_mod)
                staged = sharded.detach()
                if staged.device.type != "cpu":
                    staged = staged.cpu()
                if staged.dtype != torch.int16:
                    staged = staged.to(dtype=torch.int16)
                if not staged.is_contiguous():
                    staged = staged.contiguous()
                if _exl3_mem_waterfall_enabled():
                    _exl3_mem_snapshot("AFTER_READ", owner_mod)
                    _exl3_mem_snapshot("AFTER_LAYOUT_CONVERSION", owner_mod)
                if not hasattr(owner_mod, "_exl3_trellis_staging"):
                    owner_mod._exl3_trellis_staging = {
                        "gate": {},
                        "up": {},
                        "down": {},
                    }
                owner_mod._exl3_trellis_staging[proj][int(expert_id)] = staged
                if _exl3_mem_waterfall_enabled():
                    _exl3_mem_snapshot("AFTER_COPY_TO_FINAL", owner_mod)
                del loaded, sharded, loaded_weight
                if _exl3_mem_waterfall_enabled():
                    _exl3_mem_snapshot("AFTER_TEMP_DELETE", owner_mod)
                return True if return_success else None

            # Legacy: one independent Parameter allocation per expert.
            # Block-aligned fill (fork scheme): the reconstruct kernel's
            # 128-block Hadamard mixes channels within each block, so a
            # rank must hold the FULL block-aligned tile window spanning
            # its shard boundary — not the ragged narrow (which drops the
            # boundary block's tail tiles and decodes wrong owned
            # channels). Pre-fill zeros, then copy the real window from
            # the un-narrowed checkpoint tensor; the act-mask (stashed by
            # the suh/svh path) restricts apply to the owned channels.
            dest_device = owner_mod.w13_suh.device
            if _exl3_mem_waterfall_enabled():
                _exl3_mem_snapshot("AFTER_DEST_ALLOC", owner_mod)
            if tp_size > 1:
                if shard_id == "w2":
                    full = int(loaded.shape[0]) * 16
                else:
                    full = int(loaded.shape[1]) * 16
                if full % tp_size:
                    raise ValueError(
                        f"EXL3 MoE intermediate {full} not divisible by "
                        f"tp={tp_size}"
                    )
                real = full // tp_size
                owned_lo = tp_rank * real
                owned_hi = owned_lo + real
                lo = owned_lo // 128 * 128
                hi = -(-owned_hi // 128) * 128
                kt_lo, kt_hi = lo // 16, hi // 16
                if shard_id == "w2":
                    window = loaded.detach().contiguous()[kt_lo:kt_hi]
                else:  # w13 col path: tile range lives on dim 1
                    window = loaded.detach().contiguous()[:, kt_lo:kt_hi, :]
                # The window covers every dest tile with real data — the
                # zeros pre-fill idea is moot; the window IS the payload.
                payload = window.to(device=dest_device).contiguous()
            elif (
                sharded.dtype == torch.int16
                and sharded.device == dest_device
                and sharded.is_contiguous()
            ):
                payload = sharded
            else:
                payload = sharded.to(
                    device=dest_device, dtype=torch.int16, non_blocking=False
                ).contiguous()
            new_p = Parameter(payload, requires_grad=False)
            new_p.weight_loader = self._load_exl3
            new_p._exl3_owner = owner_mod
            plist[expert_id] = new_p
            owner_mod._exl3_trellis_alloc_count_before = int(
                getattr(owner_mod, "_exl3_trellis_alloc_count_before", 0)
            ) + 1
            del loaded, sharded, payload, loaded_weight
            if _exl3_mem_waterfall_enabled():
                _exl3_mem_snapshot("AFTER_TEMP_DELETE", owner_mod)
            return True if return_success else None

        # suh / svh remain stacked (K-independent).
        if owner_mod is None:
            raise RuntimeError("EXL3 scale load missing owner module")
        if shard_id in ("w1", "w3"):
            shard_idx = 0 if shard_id == "w1" else 1
            sharded = shard_exl3_col(loaded, suffix, tp_rank, tp_size, aligned=True)
            dest = getattr(owner_mod, "w13_" + suffix).data[expert_id, shard_idx]
        elif shard_id == "w2":
            sharded = shard_exl3_row(loaded, suffix, tp_rank, tp_size, aligned=True)
            dest = getattr(owner_mod, "w2_" + suffix).data[expert_id]
        else:
            raise ValueError(f"unknown EXL3 shard_id={shard_id}")

        if not sharded.is_contiguous():
            sharded = sharded.contiguous()
        # MoE TP sharding vs 128-block Hadamards: the reconstruct kernel mixes
        # each 128-channel block, so a shard boundary inside a block cannot be
        # served by zero-padded partial data. Fill the aligned buffer with the
        # block-aligned TRUE data (contiguous slice of the loaded tensor) and
        # mask activations to the owned channels at apply time. Stash the full
        # per-expert intermediate size on first sight.
        if tp_size > 1 and shard_id == "w2" and suffix == "suh":
            full = int(loaded.shape[0])
            if full % tp_size:
                raise ValueError(
                    f"EXL3 MoE intermediate {full} not divisible by tp={tp_size}"
                )
            owner._exl3_intermediate_full = full
        if tp_size > 1 and suffix in ("trellis", "suh", "svh"):
            full = getattr(owner, "_exl3_intermediate_full", None)
            if full is None:
                dim = 0 if shard_id == "w2" else (1 if suffix == "trellis" else 0)
                full = int(loaded.shape[dim]) * (16 if suffix == "trellis" else 1)
            real = full // tp_size
            owned_lo = tp_rank * real
            owned_hi = owned_lo + real
            lo = owned_lo // 128 * 128
            hi = -(-owned_hi // 128) * 128
            if owner is not None:
                owner._exl3_act_owned = (owned_lo - lo, owned_hi - lo)
            if suffix == "trellis":
                kt_lo, kt_hi = lo // 16, hi // 16
                if shard_id == "w2":
                    src = loaded.detach().contiguous()[kt_lo:kt_hi]
                else:  # w13 col path: sharded dim is 1 (n-tiles)
                    src = loaded.detach().contiguous()[:, kt_lo:kt_hi, :]
            elif (shard_id == "w2" and suffix == "suh") or (
                shard_id in ("w1", "w3") and suffix == "svh"
            ):
                # Channel-sharded scales: block-aligned TRUE slice.
                src = loaded.detach().contiguous()[lo:hi]
            else:  # w13 suh / w2 svh span hidden; not channel-sharded
                src = sharded
            if tuple(dest.shape) != tuple(src.shape):
                raise RuntimeError(
                    f"EXL3 aligned buffer mismatch {weight_name} shard={shard_id} "
                    f"expert={expert_id}: dest {tuple(dest.shape)} != "
                    f"src {tuple(src.shape)}"
                )
            dest.copy_(src)
            return True if return_success else None

        if tuple(dest.shape) != tuple(sharded.shape):
            # Aligned-alloc padding: create_weights allocates the 128-aligned
            # intermediate (maybe_roundup_sizes); the checkpoint ships the real
            # per-rank size. Zero-pad the single short dim — the suh/svh tail
            # rows are 0, so padded activations decode to exactly 0 and the
            # padded trellis tiles are never semantically read.
            diffs = [
                d
                for d in range(dest.dim())
                if dest.shape[d] != sharded.shape[d]
            ]
            if (
                dest.dim() == sharded.dim()
                and len(diffs) == 1
                and dest.shape[diffs[0]] > sharded.shape[diffs[0]]
            ):
                d = diffs[0]
                pad = dest.shape[d] - sharded.shape[d]
                spec = (0, 0) * (sharded.dim() - 1 - d) + (0, pad) + (0, 0) * d
                sharded = torch.nn.functional.pad(sharded, spec)
            else:
                raise RuntimeError(
                    f"EXL3 load shape mismatch {weight_name} shard={shard_id} "
                    f"expert={expert_id}: dest {tuple(dest.shape)} != "
                    f"loaded {tuple(sharded.shape)}"
                )
        dest.copy_(sharded)
        if dest.device.type == "cuda" and torch is not None:
            try:
                torch.cuda.current_stream().synchronize()
            except Exception:
                pass
        _madv_dontneed_cpu_tensor(sharded)
        del loaded, sharded, loaded_weight
        return True if return_success else None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        store = getattr(layer, "_exl3_mixed_store", None)
        if store is not None:
            layer._exl3_inners = store.build_inners(make_linear_exl3)
            layer._exl3_ptrs = None
            logger.info(
                "EXL3 tensor-mixed-K: exact-width whole experts=%d, eager reference loop; "
                "native/fused/fat dispatch disabled (GPU qualification pending)",
                store.num_experts,
            )
            return
        if not hasattr(layer, "gate_trellis"):
            return
        if not hasattr(layer, "gate_trellis"):
            return
        # Pack staged trellis tensors into contiguous per-shape arenas before
        # building LinearEXL3 handles. Views preserve exact heterogeneous K.
        # Direct-fill plans have empty staging but still need stats finalized.
        staging = getattr(layer, "_exl3_trellis_staging", None) or {}
        staged_n = sum(len(m) for m in staging.values() if isinstance(m, dict))
        has_plan = getattr(layer, "_exl3_trellis_arena_plan", None) is not None
        if _exl3_trellis_arena_enabled() and (staged_n > 0 or has_plan):
            if _exl3_mem_waterfall_enabled():
                _exl3_mem_snapshot("BEFORE_ARENA_PACK", layer)
            alloc_before = int(getattr(layer, "_exl3_trellis_alloc_count_before", 0))
            alloc_before = max(alloc_before, staged_n)
            stats = _pack_trellis_arenas(layer)
            stats["allocations_before"] = alloc_before
            layer._exl3_trellis_arena_stats = stats
            if _exl3_mem_waterfall_enabled():
                _exl3_mem_snapshot("AFTER_ARENA_PACK", layer)
                _exl3_mem_snapshot("AFTER_GC", layer)
            if not self._logged:
                logger.info(
                    "EXL3 trellis arenas: before_allocs=%s after_arenas=%s "
                    "final_bytes=%s temp_peak_bytes=%s",
                    stats.get("allocations_before"),
                    stats.get("allocations_after"),
                    stats.get("final_bytes"),
                    stats.get("temp_peak_bytes"),
                )

        # Bind owner for any late loads; stitch LinearEXL3 handles.
        for name in (
            "w13_trellis",
            "w13_suh",
            "w13_svh",
            "w13_mcg",
            "w13_mul1",
            "w2_trellis",
            "w2_suh",
            "w2_svh",
            "w2_mcg",
            "w2_mul1",
        ):
            if hasattr(layer, name):
                getattr(layer, name)._exl3_owner = layer
        for plist in (layer.gate_trellis, layer.up_trellis, layer.down_trellis):
            for param in plist:
                param._exl3_owner = layer
                param.weight_loader = self._load_exl3
        _check_moe_codebook_markers(layer.w13_mcg, layer.w13_mul1, "w13")
        _check_moe_codebook_markers(layer.w2_mcg, layer.w2_mul1, "w2")

        n_exp = int(len(layer.gate_trellis))
        inners: list[dict[str, Any]] = []
        k_values: list[int] = []
        # _BATCHED_MARKERS: prefetch every marker block once (4 host
        # transfers) instead of 6 .item() syncs per expert (288/layer).
        w13_mcg_h = _marker_row_host(layer.w13_mcg)
        w13_mul1_h = _marker_row_host(layer.w13_mul1)
        w2_mcg_h = _marker_row_host(layer.w2_mcg)
        w2_mul1_h = _marker_row_host(layer.w2_mul1)
        w13_ncol = 2  # [n_exp, 2, 1]
        for e in range(n_exp):
            gt = layer.gate_trellis[e]
            ut = layer.up_trellis[e]
            dt = layer.down_trellis[e]
            if gt.numel() == 0 or ut.numel() == 0 or dt.numel() == 0:
                raise RuntimeError(
                    f"EXL3 mixed-K load incomplete for local expert {e}: "
                    f"gate={tuple(gt.shape)} up={tuple(ut.shape)} down={tuple(dt.shape)}"
                )
            gate = make_linear_exl3(
                gt,
                layer.w13_suh[e, 0],
                layer.w13_svh[e, 0],
                _marker_tensor_or_none_host(w13_mcg_h, e * w13_ncol + 0, gt.device),
                _marker_tensor_or_none_host(w13_mul1_h, e * w13_ncol + 0, gt.device),
            )
            up = make_linear_exl3(
                ut,
                layer.w13_suh[e, 1],
                layer.w13_svh[e, 1],
                _marker_tensor_or_none_host(w13_mcg_h, e * w13_ncol + 1, ut.device),
                _marker_tensor_or_none_host(w13_mul1_h, e * w13_ncol + 1, ut.device),
            )
            down = make_linear_exl3(
                dt,
                layer.w2_suh[e],
                layer.w2_svh[e],
                _marker_tensor_or_none_host(w2_mcg_h, e, dt.device),
                _marker_tensor_or_none_host(w2_mul1_h, e, dt.device),
            )
            inners.append({"gate": gate, "up": up, "down": down})
            k_values.extend(
                [
                    int(gt.shape[-1]) // 16,
                    int(ut.shape[-1]) // 16,
                    int(dt.shape[-1]) // 16,
                ]
            )
        layer._exl3_inners = inners
        mixed_k = len(set(k_values)) > 1
        layer._exl3_mixed_k = mixed_k
        # Block-aligned buffer scheme: activations are masked to the owned
        # channel window before the down projection (see _load_exl3).
        owned = getattr(layer, "_exl3_act_owned", None)
        if owned is not None:
            layer._exl3_act_mask = torch.zeros(
                layer._exl3_intermediate_local, dtype=torch.bool,
                device=layer.w2_suh.device,
            )
            layer._exl3_act_mask[owned[0]:owned[1]] = True
            for inner in inners:
                inner["act_mask"] = layer._exl3_act_mask
        # Codebook flags (mcg, mul1) per projection for the fused kernel launch;
        # every expert in a layer must agree.
        if inners:
            flags = tuple(
                bool(getattr(inners[0][w], a, d))
                for w in ("gate", "up", "down")
                for a, d in (("mcg", True), ("mul1", False))
            )
            for e, inner in enumerate(inners):
                f_e = tuple(
                    bool(getattr(inner[w], a, d))
                    for w in ("gate", "up", "down")
                    for a, d in (("mcg", True), ("mul1", False))
                )
                if f_e != flags:
                    raise RuntimeError(
                        f"EXL3 experts disagree on codebook: expert {e} {f_e} vs expert 0 {flags}"
                    )
            layer._exl3_codebook_flags = flags
        fused_ok = False
        fused_err = None
        # Fused/native MoE launches take a single K for gate/up/down across the
        # whole layer. Heterogeneous packed K uses exl3_moe_mixedk when available,
        # otherwise falls back to the LinearEXL3 loop.
        backend = get_moe_kernel_backend()
        if mixed_k:
            # Check for the mixed-K fused kernel in exllamav3_ext.
            try:
                _ext = load_exllamav3_ext()
                has_mixedk = _ext is not None and hasattr(_ext, "exl3_moe_mixedk")
            except Exception:
                has_mixedk = False
            if has_mixedk and (fused_moe_enabled() or backend == "native"):
                try:
                    build_exl3_fused_state(layer, inners)
                    fused_ok = True
                except Exception as exc:
                    fused_err = repr(exc)
                    layer._exl3_ptrs = None
                    layer._exl3_mixedk_unified = False
            else:
                fused_err = f"mixed_packed_K={sorted(set(k_values))}"
                if not has_mixedk:
                    fused_err += " (exl3_moe_mixedk not available)"
                layer._exl3_ptrs = None
                layer._exl3_fused_temps = None
                layer._exl3_fused_concurrency = 0
                layer._exl3_mixedk_unified = False
        elif fused_moe_enabled() or backend == "native":
            try:
                has_native = backend == "native" and native_moe_kernel_available()
                has_exllamav3 = _exllamav3_moe_available()
                if has_native or has_exllamav3:
                    build_exl3_fused_state(layer, inners)
                    fused_ok = True
                else:
                    fused_err = "no native or exllamav3 MoE kernel available"
            except Exception as exc:
                fused_err = repr(exc)
                layer._exl3_ptrs = None
        if not self._logged and self.bits != self.quant_config.bits:
            logger.info(
                "EXL3 per-layer K override: layer prefix %s uses bits=%d (base %d)",
                getattr(layer, "layer_name", None) or getattr(layer, "prefix", "?"),
                self.bits,
                self.quant_config.bits,
            )
        if not self._logged:
            if fused_ok:
                fused_label = "exl3_moe_mixedk" if getattr(layer, "_exl3_mixedk_unified", False) else "exl3_moe"
                logger.info(
                    "EXL3 MCG trellis engaged for routed experts: bits=%s "
                    "experts_local=%s hidden=%s intermediate_local=%s "
                    "fused_moe=%s concurrency=%s "
                    "(no BF16 expert reconstruct at load)",
                    self.bits,
                    n_exp,
                    layer._exl3_hidden_size,
                    layer._exl3_intermediate_local,
                    fused_label,
                    getattr(layer, "_exl3_fused_concurrency", "?"),
                )
            else:
                logger.info(
                    "EXL3 MCG trellis engaged for routed experts: bits=%s "
                    "experts_local=%s hidden=%s intermediate_local=%s "
                    "fused_moe=python_loop (%s) mixed_k=%s "
                    "(no BF16 expert reconstruct at load)",
                    self.bits,
                    n_exp,
                    layer._exl3_hidden_size,
                    layer._exl3_intermediate_local,
                    fused_err or "EXL3_FUSED_MOE=0",
                    mixed_k,
                )
            self._logged = True
        # Release transient load leftovers between MoE layers on UMA hosts.
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        raw_limit = getattr(self.moe, "swiglu_limit", None)
        try:
            parsed_limit = float(raw_limit)
        except (TypeError, ValueError, OverflowError):
            parsed_limit = None
        limit = (
            parsed_limit
            if parsed_limit is not None
            and math.isfinite(parsed_limit)
            and parsed_limit > 0
            else None
        )
        return apply_exl3_experts(
            x, topk_ids, topk_weights, layer, limit=limit
        )


# ---------------------------------------------------------------------------
# Row-wise EXL3 embedding tables (exllamav3 n-gram format)
# ---------------------------------------------------------------------------

NGRAM_ROW_DIM = 160
NGRAM_MUL1 = 0x83DCD12D


def ngram_words_per_row(bits: int) -> int:
    """Packed int16 words per row: one fp16 scale word plus the K-bit ring bitstream."""
    return 1 + NGRAM_ROW_DIM * int(bits) // 16


def _derive_ngram_embedding_spec(
    tensor_storage: dict[str, Any],
) -> dict[str, Any] | None:
    """Derive the ngram_embedding spec from a tensor_storage ledger.

    Some packs (Qwen3.8-Flash-Next) ledger the packed n-gram table under
    tensor_storage but omit the top-level ngram_embedding spec. Reconstruct
    the spec from the ledgered trellis shape: rows = trellis rows,
    words = trellis cols, bits = the K whose packed width matches
    ngram_words_per_row, num_heads = the head_bias leading dim, and
    num_shards = 1 (the ledger holds a single packed table).
    """
    # Ledger layouts differ: either a flat {name: entry} map or a per-module
    # {module: {"stored_tensors": {name: entry}}} nesting.
    flat: dict[str, Any] = {}
    for key, entry in tensor_storage.items():
        stored = entry.get("stored_tensors") if isinstance(entry, dict) else None
        if stored:
            flat.update(stored)
        else:
            flat[key] = entry
    prefix = None
    trellis = None
    for name, entry in flat.items():
        if not name.endswith(".ngram_embedding.trellis"):
            continue
        if prefix is not None and not name.startswith(prefix):
            # Multiple distinct tables: ambiguous without an explicit spec.
            return None
        prefix = name.rsplit(".trellis", 1)[0]
        trellis = entry
    if trellis is None:
        return None
    shape = trellis.get("shape") or []
    if len(shape) != 2:
        return None
    rows, words = int(shape[0]), int(shape[1])
    bits = next((k for k in range(1, 9) if ngram_words_per_row(k) == words), None)
    if bits is None:
        return None
    head_bias = flat.get(f"{prefix}.head_bias")
    head_shape = (head_bias or {}).get("shape") or []
    if len(head_shape) != 2 or int(head_shape[1]) != NGRAM_ROW_DIM:
        return None
    return {
        "bits": bits,
        "num_shards": 1,
        "rows_per_shard": rows,
        "num_heads": int(head_shape[0]),
        "modules": ["ngram_embedding"],
    }


def _fp16_from_bits(bits16: int, device) -> torch.Tensor:
    signed = bits16 - 0x10000 if bits16 >= 0x8000 else bits16
    return torch.tensor([signed], dtype=torch.int16, device=device).view(torch.float16)


# Reimplements the mul1 codebook arithmetic of ExLlamaV3's ngram_codec
# (Copyright (c) 2025 Turboderp, MIT); see THIRD_PARTY_NOTICES.md.
def ngram_mul1_codebook(device) -> torch.Tensor:
    """The 65536-entry mul1 codebook as fp16, as exllamav3's cached table."""
    state = torch.arange(65536, device=device, dtype=torch.int64)
    prod = (state * NGRAM_MUL1) & 0xFFFFFFFF
    h = 1024.0 + (
        (prod & 0xFF) + ((prod >> 8) & 0xFF) + ((prod >> 16) & 0xFF) + ((prod >> 24) & 0xFF)
    ).to(torch.float32)
    k_inv = _fp16_from_bits(0x1EEE, device).to(torch.float32)
    k_bias = _fp16_from_bits(0xC931, device).to(torch.float32)
    return (h * k_inv + k_bias).to(torch.float16)


def _ngram_bit_tables(bits: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(word index, bit index) of stream bit m of element i.

    Bit m of element i lives at ring position ((i - m // K) mod ROW_DIM) * K + m % K,
    offset by one for the scale word.
    """
    i = torch.arange(NGRAM_ROW_DIM, device=device, dtype=torch.int64).unsqueeze(1)
    m = torch.arange(16, device=device, dtype=torch.int64).unsqueeze(0)
    pos = (i - m // bits) % NGRAM_ROW_DIM
    sb = pos * bits + m % bits
    return 1 + (sb >> 4), sb & 15


# Reimplements the row layout of ExLlamaV3's ngram_codec / ngram_dequant
# kernel (Copyright (c) 2025 Turboderp, MIT); see THIRD_PARTY_NOTICES.md.
def ngram_dequant_rows_torch(
    packed: torch.Tensor,
    bits: int,
    heads: torch.Tensor,
    head_bias: torch.Tensor,
    chunk: int = 8192,
) -> torch.Tensor:
    """Pure-torch twin of ``exllamav3_ext.ngram_dequant`` (fallback and test oracle).

    packed: (N, words) int16; heads: (N,) int; head_bias: (num_heads, ROW_DIM) fp16.
    Returns (N, ROW_DIM) fp16: codebook[state] * scale + head_bias[head].
    """
    device = packed.device
    widx, bidx = _ngram_bit_tables(bits, device)
    codebook = ngram_mul1_codebook(device)
    shifts = torch.arange(16, device=device, dtype=torch.int64)
    out = torch.empty(packed.shape[0], NGRAM_ROW_DIM, dtype=torch.float16, device=device)
    for s in range(0, packed.shape[0], chunk):
        p = packed[s : s + chunk]
        scale = p[:, 0].contiguous().view(torch.float16).to(torch.float32)
        words = (p.to(torch.int64) & 0xFFFF)[:, widx]
        state = (((words >> bidx) & 1) << shifts).sum(-1)
        vals = codebook[state].to(torch.float32)
        bias = head_bias[heads[s : s + chunk].to(torch.int64)].to(torch.float32)
        out[s : s + chunk] = (vals * scale.unsqueeze(1) + bias).to(torch.float16)
    return out


NGRAM_TABLE_ENV = "VLLM_EXL3_NGRAM_TABLE"


class _NgramDiskTable:
    """The packed n-gram table as the checkpoint's own CPU views, one per shard.
    Rows are gathered on the host; with memory-mapped views that is a page-cache
    read, so the table costs no device memory and no anonymous RAM."""

    def __init__(self, views: list[torch.Tensor], rows_per_shard: int) -> None:
        self.views = views
        self.rows_per_shard = int(rows_per_shard)
        self.num_rows = sum(int(v.shape[0]) for v in views)

    def gather(self, uids_cpu: torch.Tensor) -> torch.Tensor:
        if len(self.views) == 1:
            return self.views[0].index_select(0, uids_cpu)
        shard = uids_cpu // self.rows_per_shard
        local = uids_cpu - shard * self.rows_per_shard
        out = torch.empty((uids_cpu.numel(), self.views[0].shape[1]), dtype=self.views[0].dtype)
        for s in shard.unique().tolist():
            m = shard == s
            out[m] = self.views[s].index_select(0, local[m])
        return out


def _ngram_view_owner(param: Parameter) -> torch.nn.Module:
    """The embedding layer a disk-mode shard parameter belongs to (set at create time)."""
    owner = getattr(param, "_exl3_ngram_owner", None)
    if owner is None:
        raise RuntimeError("EXL3 n-gram (disk): shard parameter has no owning layer")
    return owner


def _check_ngram_disk_graph_mode() -> None:
    """Disk mode synchronizes with the host inside the model forward; refuse the CUDA
    graph modes that would capture that, and say what to pass instead."""
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config().compilation_config
    except Exception:
        return
    mode = getattr(cfg, "cudagraph_mode", None)
    name = getattr(mode, "name", str(mode))
    ops = list(getattr(cfg, "splitting_ops", None) or [])
    if "FULL" in name:
        raise RuntimeError(
            f"{NGRAM_TABLE_ENV}=disk needs PIECEWISE CUDA graphs with the lookup kept "
            "eager; got cudagraph_mode=%s. Pass --compilation-config with "
            '{"cudagraph_mode": "PIECEWISE", "splitting_ops": [<the attention ops>, '
            '"vllm::exl3_ngram_lookup_out"]}' % name
        )
    if ops and "vllm::exl3_ngram_lookup_out" not in ops:
        raise RuntimeError(
            f"{NGRAM_TABLE_ENV}=disk: add \"vllm::exl3_ngram_lookup_out\" to splitting_ops so "
            "the host gather runs outside the piecewise graphs"
        )


class Exl3EmbeddingMethod(QuantizeMethodBase):
    """Row-wise EXL3 embedding table in exllamav3's n-gram format.

    Each row is stored packed: word 0 holds the row's fp16 scale, the remaining
    ROW_DIM * K / 16 int16 words hold a tail-biting ring bitstream of 160 K-bit
    trellis states. A lookup gathers packed rows and decodes them on the fly
    (mul1 codebook * scale + per-head bias) into fp16, so the table stays at K
    bits per weight in device memory (32.6 GB for the Qwen3.8-Flash-Next table
    instead of 102 GB as bf16). Checkpoint layout under the table prefix:
    ``shard_<i>.trellis`` int16 [rows_per_shard, words], ``head_bias`` fp16
    [heads, 160], ``head_offsets`` / ``head_vocab_sizes`` int64 [heads],
    ``layer_multipliers`` int64 [n]. Shard parameters are registered as child
    modules so vLLM's AutoWeightsLoader lands them by name; they alias one
    contiguous table used for the gather.
    """

    def __init__(self, quant_config: Exl3Config, spec: dict[str, Any]) -> None:
        self.quant_config = quant_config
        self.bits = int(spec["bits"])
        self.num_shards = int(spec["num_shards"])
        self.rows_per_shard = int(spec["rows_per_shard"])
        self.num_heads = int(spec["num_heads"])
        self.words = ngram_words_per_row(self.bits)
        # Checkpoint layout: ``shard_<i>.trellis`` (default) or one ``trellis`` tensor
        # holding the whole table (the layout exllamav3 1.5.0-era packs ship).
        self.sharded = bool(spec.get("sharded", True))
        if not self.sharded and self.num_shards != 1:
            raise ValueError(
                "ngram_embedding: an unsharded table must declare num_shards=1, "
                f"got {self.num_shards}"
            )
        kernel = os.environ.get("VLLM_EXL3_NGRAM_KERNEL", "ext").strip().lower()
        if kernel not in ("ext", "torch"):
            raise ValueError(
                f"VLLM_EXL3_NGRAM_KERNEL must be 'ext' or 'torch', got {kernel!r}"
            )
        self.kernel = kernel
        # Where the packed table lives. ``resident``: one int16 device tensor (32.6 GiB
        # for the Qwen3.8-Flash-Next 5-bit table). ``disk``: the loader keeps the
        # checkpoint's memory-mapped views and every lookup gathers the rows it needs on
        # the host, so the table costs page cache, not device memory. See
        # ``_embedding_impl_disk`` for what that requires of the CUDA graph mode.
        table_mode = os.environ.get(NGRAM_TABLE_ENV, "resident").strip().lower()
        if table_mode not in ("resident", "disk"):
            raise ValueError(f"{NGRAM_TABLE_ENV} must be 'resident' or 'disk', got {table_mode!r}")
        self.table_mode = table_mode
        self._ext = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, extra_weight_attrs
        if int(input_size_per_partition) != NGRAM_ROW_DIM:
            raise ValueError(
                f"EXL3 n-gram rows are {NGRAM_ROW_DIM} wide; layer asks for "
                f"{int(input_size_per_partition)}"
            )
        total_rows = self.num_shards * self.rows_per_shard
        parts = [int(s) for s in output_partition_sizes]
        if int(output_size) != total_rows or parts != [total_rows]:
            raise ValueError(
                "EXL3 n-gram table geometry mismatch: vLLM built "
                f"{int(output_size)} rows (partitions {parts}) but the checkpoint "
                f"holds {self.num_shards} shards x {self.rows_per_shard} rows = {total_rows}"
            )
        _, tp_size = _resolve_tp_geometry(layer)
        if int(tp_size) != 1:
            raise RuntimeError("EXL3 n-gram embedding supports tensor parallel size 1 only")

        if self.table_mode == "resident":
            table = torch.empty(
                self.num_shards, self.rows_per_shard, self.words, dtype=torch.int16
            )
        else:
            # Nothing resident: the loader keeps the checkpoint views (``_exl3_ngram_views``)
            # and the parameters below are name anchors for vLLM's weight loader only.
            table = None
            layer._exl3_ngram_views = [None] * self.num_shards
        loaded: set[int] = set()
        for i in range(self.num_shards):
            data = table[i] if table is not None else torch.empty(0, dtype=torch.int16)
            p = Parameter(data, requires_grad=False)
            p.weight_loader = self._make_shard_loader(i, loaded)
            if table is None:
                p._exl3_ngram_owner = layer
            if self.sharded:
                shard = torch.nn.Module()
                shard.register_parameter("trellis", p)
                layer.add_module(f"shard_{i}", shard)
            else:
                layer.register_parameter("trellis", p)
        aux = {
            "head_bias": Parameter(
                torch.zeros(self.num_heads, NGRAM_ROW_DIM, dtype=torch.float16),
                requires_grad=False,
            ),
            "head_offsets": Parameter(
                torch.full((self.num_heads,), -1, dtype=torch.int64), requires_grad=False
            ),
            "head_vocab_sizes": Parameter(
                torch.zeros(self.num_heads, dtype=torch.int64), requires_grad=False
            ),
            "layer_multipliers": Parameter(
                torch.zeros(0, dtype=torch.int64), requires_grad=False
            ),
        }
        aux_loaded: set[str] = set()
        for name, p in aux.items():
            p.weight_loader = self._make_aux_loader(name, aux_loaded)
            layer.register_parameter(name, p)
        layer._exl3_ngram_table = table
        layer._exl3_ngram_loaded = loaded
        layer._exl3_ngram_aux_loaded = aux_loaded
        layer._exl3_ngram_dtype = params_dtype

    def _make_shard_loader(self, index: int, loaded: set[int]):
        rows, words, bits = self.rows_per_shard, self.words, self.bits
        disk = self.table_mode == "disk"

        def weight_loader(param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id=None):
            del loaded_shard_id
            if loaded_weight.dtype != torch.int16 or tuple(loaded_weight.shape) != (rows, words):
                raise ValueError(
                    f"EXL3 n-gram shard {index}: expected int16 ({rows}, {words}) for "
                    f"K={bits}, got {loaded_weight.dtype} {tuple(loaded_weight.shape)}"
                )
            if disk:
                # vLLM's safetensors iterator hands over a zero-copy view of the mapped
                # file; holding it keeps the mapping alive and no row is read until a
                # lookup touches it. A tensor that is not a plain CPU view (a loader
                # that copied, or another device) is kept as-is and still works.
                owner = _ngram_view_owner(param)
                owner._exl3_ngram_views[index] = loaded_weight.detach()
            else:
                param.data.copy_(loaded_weight)
            loaded.add(index)

        return weight_loader

    def _make_aux_loader(self, name: str, aux_loaded: set[str]):
        def weight_loader(param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id=None):
            del loaded_shard_id
            if name == "layer_multipliers":
                param.data = loaded_weight.to(device=param.device, dtype=param.dtype).clone()
            else:
                if tuple(loaded_weight.shape) != tuple(param.shape):
                    raise ValueError(
                        f"EXL3 n-gram {name}: expected shape {tuple(param.shape)}, "
                        f"got {tuple(loaded_weight.shape)}"
                    )
                param.data.copy_(loaded_weight.to(dtype=param.dtype))
            aux_loaded.add(name)

        return weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        table = getattr(layer, "_exl3_ngram_table", None)
        if table is None and not hasattr(layer, "_exl3_ngram_views"):
            return
        loaded = layer._exl3_ngram_loaded
        missing = [i for i in range(self.num_shards) if i not in loaded]
        if missing:
            raise RuntimeError(
                f"EXL3 n-gram table: {len(missing)} of {self.num_shards} shards never "
                f"loaded (first missing: {missing[:8]})"
            )
        aux_missing = [
            n for n in ("head_bias", "head_offsets", "head_vocab_sizes")
            if n not in layer._exl3_ngram_aux_loaded
        ]
        if aux_missing:
            raise RuntimeError(f"EXL3 n-gram table: aux tensors never loaded: {aux_missing}")
        first = layer.shard_0.trellis if self.sharded else layer.trellis
        if table is not None and first.data_ptr() != table.data_ptr():
            raise RuntimeError(
                "EXL3 n-gram shard parameters no longer alias the packed table; refusing to serve"
            )
        offs = layer.head_offsets.detach().cpu().tolist()
        sizes = layer.head_vocab_sizes.detach().cpu().tolist()
        total_rows = self.num_shards * self.rows_per_shard
        consistent = (
            offs[0] == 0
            and all(offs[i + 1] == offs[i] + sizes[i] for i in range(len(offs) - 1))
            and offs[-1] + sizes[-1] <= total_rows
        )
        if not consistent:
            raise RuntimeError(
                f"EXL3 n-gram head layout inconsistent with the table: offsets={offs} "
                f"sizes={sizes} rows={total_rows}"
            )
        if table is not None:
            layer._exl3_ngram_rows = table.view(-1, self.words)
        else:
            views = layer._exl3_ngram_views
            if any(v is None for v in views):
                raise RuntimeError("EXL3 n-gram table (disk): a shard view was never captured")
            layer._exl3_ngram_rows = None
            layer._exl3_ngram_disk = _NgramDiskTable(views, self.rows_per_shard)
            layer._exl3_ngram_head_offsets_cpu = layer.head_offsets.data.detach().cpu().contiguous()
            _check_ngram_disk_graph_mode()
        layer._exl3_ngram_head_offsets = layer.head_offsets.data.contiguous()
        layer._exl3_ngram_head_bias = layer.head_bias.data.contiguous()
        layer._exl3_opaque_name = _exl3_register_opaque_layer(layer, "ngram")
        if self.kernel == "ext":
            ext = load_exllamav3_ext()
            if hasattr(ext, "ngram_dequant"):
                self._ext = ext
            else:
                logger.warning(
                    "exllamav3_ext has no ngram_dequant; the EXL3 n-gram table falls back "
                    "to the torch decoder"
                )
                self.kernel = "torch"
        logger.info(
            "EXL3 n-gram embedding ready: %d shards x %d rows (%s), K=%d, %d heads, "
            "%.2f GiB packed, table=%s, kernel=%s",
            self.num_shards, self.rows_per_shard,
            "sharded" if self.sharded else "unsharded", self.bits, self.num_heads,
            self.num_shards * self.rows_per_shard * self.words * 2 / 2**30,
            self.table_mode, self.kernel,
        )

    def _lookup_packed(self, layer: torch.nn.Module, ids_flat: torch.Tensor) -> torch.Tensor:
        return layer._exl3_ngram_rows.index_select(0, ids_flat)

    def _heads_for(self, layer: torch.nn.Module, ids_flat: torch.Tensor) -> torch.Tensor:
        found = torch.searchsorted(layer._exl3_ngram_head_offsets, ids_flat, right=True) - 1
        return found.clamp_(0, self.num_heads - 1).to(torch.int32)

    def _decode(self, layer: torch.nn.Module, packed: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
        bias = layer._exl3_ngram_head_bias
        if self.kernel == "ext" and self._ext is not None and packed.is_cuda:
            out = torch.empty(
                packed.shape[0], NGRAM_ROW_DIM, dtype=torch.float16, device=packed.device
            )
            self._ext.ngram_dequant(packed, self.bits, heads, bias, out)
            return out
        return ngram_dequant_rows_torch(packed, self.bits, heads, bias)

    def _ngram_lookup_uses_out_variant(self, layer: torch.nn.Module) -> bool:
        """Whether this lookup has to write into a caller-allocated buffer.

        Only the opt-in disk table needs it. There the lookup runs eagerly as a
        CUDA-graph splitting op, so the output buffer must be allocated in the
        piece before the split. The resident table keeps the mainline returning
        op, so nothing about the default path changes.
        """
        return getattr(layer, "_exl3_ngram_disk", None) is not None

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        name = getattr(layer, "_exl3_opaque_name", None)
        if name is not None and _EXL3_OPS_READY:
            if not self._ngram_lookup_uses_out_variant(layer):
                return torch.ops.vllm.exl3_ngram_lookup(input_, name)
            # Out-variant on purpose, and reachable only with
            # ``VLLM_EXL3_NGRAM_TABLE=disk``. When this op is a splitting op (disk
            # mode), the piecewise CUDA graph after it was captured reading its
            # input at one address; a fresh tensor returned from an eager op lands
            # anywhere. The buffer is allocated here, inside the piece before the
            # split, so its address is the graph's own and stable across replays,
            # the same way vLLM's attention and PLE ops take their output as an
            # argument.
            out = torch.empty(
                *input_.shape, NGRAM_ROW_DIM, dtype=layer._exl3_ngram_dtype, device=input_.device
            )
            torch.ops.vllm.exl3_ngram_lookup_out(input_, name, out)
            return out
        # The custom op registers only the platform dispatch key (CUDA); the
        # PLE offload worker runs this lookup on CPU (the packed table lives
        # in host RAM), so route non-CUDA eager calls to the impl directly.
        if name is not None and _EXL3_OPS_READY and input_.is_cuda:
            return torch.ops.vllm.exl3_ngram_lookup(input_, name)
        return self._embedding_impl(layer, input_)

    def _embedding_impl(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        if getattr(layer, "_exl3_ngram_disk", None) is not None:
            return self._embedding_impl_disk(layer, input_)
        if getattr(layer, "_exl3_ngram_rows", None) is None:
            raise RuntimeError("EXL3 n-gram table was not finalized after weight load")
        ids = input_.reshape(-1).to(torch.int64)
        packed = self._lookup_packed(layer, ids)
        heads = self._heads_for(layer, ids)
        out = self._decode(layer, packed, heads)
        return out.to(layer._exl3_ngram_dtype).view(*input_.shape, NGRAM_ROW_DIM)

    def _embedding_impl_disk(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        """Host-gathered lookup: unique row ids to the CPU, rows from the mapped
        checkpoint, one upload, decode on the device, expand back.

        The device-to-host copy is a synchronization point, so this op must run
        eagerly: PIECEWISE CUDA graphs with ``vllm::exl3_ngram_lookup_out`` in
        ``splitting_ops``. Under a FULL graph the copy cannot be captured.
        """
        ids = input_.reshape(-1).to(torch.int64)
        uids, inverse = torch.unique(ids, return_inverse=True)
        uids_cpu = uids.to("cpu", torch.int64)
        packed_cpu = layer._exl3_ngram_disk.gather(uids_cpu)
        heads_cpu = torch.searchsorted(
            layer._exl3_ngram_head_offsets_cpu, uids_cpu, right=True
        ) - 1
        heads_cpu = heads_cpu.clamp_(0, self.num_heads - 1).to(torch.int32)
        # Blocking uploads on purpose: a non_blocking copy out of a pinned temporary
        # can outlive the temporary and read freed memory, and the device-to-host copy
        # above already synchronized this stream, so nothing is gained by overlapping.
        packed = packed_cpu.to(input_.device)
        heads = heads_cpu.to(input_.device)
        rows = self._decode(layer, packed, heads)
        out = rows.index_select(0, inverse.to(rows.device))
        return out.to(layer._exl3_ngram_dtype).view(*input_.shape, NGRAM_ROW_DIM)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None):
        raise NotImplementedError("EXL3 n-gram tables only support embedding lookup")


# ---------------------------------------------------------------------------
# torch.compile opacity: vLLM traces the model forward with fullgraph dynamo, which
# cannot step into exllamav3's pybind kernels. The dense linear forward and the
# n-gram lookup run behind vLLM custom ops, looked up by a stable layer name.
# ---------------------------------------------------------------------------

_EXL3_OPAQUE_LAYERS: dict[str, Any] = {}
_EXL3_OPS_READY = False


def _exl3_register_opaque_layer(layer: torch.nn.Module, kind: str) -> str:
    stable = (
        getattr(layer, "_exl3_prefix", None)
        or getattr(layer, "prefix", None)
        or getattr(layer, "layer_name", None)
        or f"id{id(layer)}"
    )
    name = f"exl3_{kind}:{stable}"
    _EXL3_OPAQUE_LAYERS[name] = layer
    return name


def _exl3_linear_forward_op(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    layer = _EXL3_OPAQUE_LAYERS[layer_name]
    return layer.quant_method._apply_impl(layer, x)


def _exl3_linear_forward_fake(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    layer = _EXL3_OPAQUE_LAYERS[layer_name]
    out = int(
        sum(
            getattr(
                layer, "_exl3_linear_true_out", layer._exl3_linear_output_partition_sizes
            )
        )
    )
    return x.new_empty(*x.shape[:-1], out)


def _exl3_ngram_lookup_op(ids: torch.Tensor, layer_name: str) -> torch.Tensor:
    layer = _EXL3_OPAQUE_LAYERS[layer_name]
    return layer.quant_method._embedding_impl(layer, ids)


def _exl3_ngram_lookup_fake(ids: torch.Tensor, layer_name: str) -> torch.Tensor:
    layer = _EXL3_OPAQUE_LAYERS[layer_name]
    return ids.new_empty(*ids.shape, NGRAM_ROW_DIM, dtype=layer._exl3_ngram_dtype)


def _exl3_ngram_lookup_out_op(ids: torch.Tensor, layer_name: str, out: torch.Tensor) -> None:
    layer = _EXL3_OPAQUE_LAYERS[layer_name]
    out.copy_(layer.quant_method._embedding_impl(layer, ids))


def _exl3_ngram_lookup_out_fake(ids: torch.Tensor, layer_name: str, out: torch.Tensor) -> None:
    return None


def _exl3_register_custom_ops() -> bool:
    global _EXL3_OPS_READY
    if _EXL3_OPS_READY:
        return True
    try:
        try:
            from vllm.utils.torch_utils import direct_register_custom_op
        except ImportError:
            from vllm.utils import direct_register_custom_op
    except ImportError:
        return False
    try:
        if not hasattr(torch.ops.vllm, "exl3_linear_forward"):
            direct_register_custom_op(
                op_name="exl3_linear_forward",
                op_func=_exl3_linear_forward_op,
                mutates_args=[],
                fake_impl=_exl3_linear_forward_fake,
            )
        if not hasattr(torch.ops.vllm, "exl3_ngram_lookup"):
            direct_register_custom_op(
                op_name="exl3_ngram_lookup",
                op_func=_exl3_ngram_lookup_op,
                mutates_args=[],
                fake_impl=_exl3_ngram_lookup_fake,
            )
        if not hasattr(torch.ops.vllm, "exl3_ngram_lookup_out"):
            direct_register_custom_op(
                op_name="exl3_ngram_lookup_out",
                op_func=_exl3_ngram_lookup_out_op,
                mutates_args=["out"],
                fake_impl=_exl3_ngram_lookup_out_fake,
            )
    except Exception as exc:  # pragma: no cover - registration is best effort
        logger.warning("EXL3 custom op registration failed; eager fallback: %r", exc)
        return False
    _EXL3_OPS_READY = True
    return True


if _VLLM_AVAILABLE and _TORCH_AVAILABLE:
    _exl3_register_custom_ops()



def _env_prefill_sync_rows() -> int:
    raw = os.environ.get("VLLM_EXL3_PREFILL_SYNC", "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 256


_EXL3_PREFILL_SYNC = _env_prefill_sync_rows()


def _prefill_sync(rows: int) -> None:
    """Workaround for the nightly V2 runner wedge on 33..144-row prefills: serialize the
    CPU against the device before each EXL3 kernel call of a prefill step. Off unless
    VLLM_EXL3_PREFILL_SYNC=<max_rows> is set; never inside CUDA graph capture; never for
    single-row (decode) calls, so decode speed is unchanged."""
    if 1 < rows <= _EXL3_PREFILL_SYNC and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize()


_EXL3_GEMV_MAX_ROWS = 2
_EXL3_RECONSTRUCT_THRESHOLD = 144


def _env_int(name: str, default: int) -> int:
    """Return an int from the environment with a fallback."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw, 10)
    except (ValueError, TypeError):
        return default


_EXL3_RECON_MIN_ROWS = _env_int("VLLM_EXL3_RECONSTRUCT_MIN_ROWS", 17)
_EXL3_COOP_GEMM = os.environ.get("VLLM_EXL3_COOP_GEMM", "").strip() in ("1", "true", "yes")


def _dense_forward(linear, x_fp16: torch.Tensor) -> torch.Tensor:
    """Dense EXL3 forward with the wedge-prone row range kept off the coop GEMM."""
    rows = int(x_fp16.shape[0])
    if (not _EXL3_COOP_GEMM) and _EXL3_RECON_MIN_ROWS <= rows <= _EXL3_RECONSTRUCT_THRESHOLD:
        out = linear.forward(x_fp16, {"reconstruct": True}, out_dtype=torch.float32)
    else:
        out = linear.forward(x_fp16, {}, out_dtype=torch.float32)
    return out



class Exl3LinearMethod(LinearMethodBase):
    """Non-routed (dense) EXL3 linear method for QKV/MLP dense projections.

    This method handles trellis/suh/svh/mcg parameters for non-routed dense
    linear layers, building LinearEXL3 objects after weight loading and applying
    them with proper TP slicing and shard concatenation.
    """

    def __init__(self, quant_config: Exl3Config, bits: int | None = None) -> None:
        self.quant_config = quant_config
        self.bits = int(bits) if bits is not None else quant_config.bits
        self._logged = False

    def create_weights(
        self,
        layer,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear,
            RowParallelLinear,
            QKVParallelLinear,
            MergedColumnParallelLinear,
        )
        from vllm.distributed import get_tensor_model_parallel_world_size

        # Determine layer type and shard behavior
        n_shards = len(output_partition_sizes)
        is_row_parallel = isinstance(layer, RowParallelLinear)
        is_col_parallel = isinstance(layer, ColumnParallelLinear)
        is_qkv_parallel = isinstance(layer, QKVParallelLinear)
        is_merged_col_parallel = isinstance(layer, MergedColumnParallelLinear)

        # For column-parallel: output dimension is sharded (each shard has different outputs)
        # For row-parallel: input dimension is sharded (each shard has same input, different outputs)
        if is_row_parallel:
            # Input is partitioned across ranks, each rank gets full input height
            in_per_partition = input_size_per_partition
        else:
            # Column-parallel or unsharded: each rank gets full input
            in_per_partition = input_size_per_partition

        # Get bf16_shards from config (may be empty)
        bf16_shards = self.quant_config._bf16_shards_for(getattr(layer, "prefix", ""))
        if bf16_shards:
            _, tp_size = _resolve_tp_geometry(layer)
            if tp_size > 1:
                raise RuntimeError(
                    f"EXL3 bf16 shards are not supported with TP size > 1; tp_size={tp_size}"
                )

        # Batched (bmm) layers: the model registers prefixes whose
        # checkpoint tensors are per-slice (slice.N.*). The slice
        # count (per rank) comes from Exl3Config.bmm_prefixes — set
        # by the model before layer construction (layer attrs set
        # after construction are too late: create_weights runs inside
        # the constructor).
        bmm_slices = 0
        _prefix = getattr(layer, "prefix", "") or ""
        _bmm_map = getattr(self, "bmm_prefixes", None) or getattr(
            self.quant_config, "bmm_prefixes", {}
        )
        for _pat, _n in _bmm_map.items():
            if _prefix.endswith(_pat):
                bmm_slices = int(_n)
                break
        is_bmm = bmm_slices > 1 or bool(getattr(layer, "is_bmm", False))
        layer._exl3_linear_is_bmm = is_bmm
        layer._exl3_bmm_slices = bmm_slices
        # K words per shard. Mixed-K merged groups (per-tensor calibration
        # packs) resolve to a per-shard K list; the trellis below allocates
        # each shard's own width and apply dispatches per shard K.
        shard_ks: list[int] | None = None
        resolved = self.quant_config._resolve_prefix_bits_from_checkpoint(
            _prefix
        )
        if isinstance(resolved, list):
            # The resolver returns one K per CHECKPOINT TENSOR in mapping
            # order. Expand to per-shard Ks using the model's shard spans
            # (in_proj_qkv covers q,k,v = 3 shards; in_proj_z covers 1).
            # Spans mirror the resolver's merged_splits table; the sum
            # must equal the shard count or the layout is unknown — fall
            # back to the heuristic rather than guess.
            _span_table = {
                "gate_up_proj": (1, 1),
                "qkv_proj": (1, 1, 1),
                "in_proj_qkvz": (3, 1),
                "kv_proj": (1, 1),
            }
            _module = _prefix.rsplit(".", 1)[-1]
            _spans = _span_table.get(_module)
            if _spans is not None and sum(_spans) == n_shards and len(
                    resolved) == len(_spans):
                shard_ks = [
                    k for k, span in zip(resolved, _spans)
                    for _ in range(span)
                ]
        k_words = (
            (max(shard_ks) if shard_ks else self.bits) * 16
        )
        if shard_ks:
            layer._exl3_shard_ks = shard_ks
        # EXL3 pads both matrix dims to multiples of 128 (zeros at the end;
        # padded output columns carry svh = 0, padded input rows only see
        # zero-extended inputs). Allocate the padded geometry, load the
        # checkpoint tensors whole, and pad / trim activations in apply().
        true_in = int(in_per_partition)
        true_out_sizes = [int(s) for s in output_partition_sizes]
        in_per_partition = _exl3_pad128(true_in)
        output_partition_sizes = [_exl3_pad128(s) for s in true_out_sizes]
        padded = in_per_partition != true_in or output_partition_sizes != true_out_sizes
        # Tile-aligned padding (pad < 16 rows/cols) is safe under TP:
        # the padded region is zeroed below and never contributes to
        # the output (padded svh rows are zero). Larger pads remain
        # unsupported.
        _pad_rows = sum(output_partition_sizes) - sum(true_out_sizes)
        _pad_in = in_per_partition - true_in
        # Tile-aligned: the pad is a whole number of 16-wide tiles —
        # the padded region decodes as complete zero tiles.
        _tile_aligned_pad = padded and _pad_rows % 16 == 0 and _pad_in % 16 == 0
        if padded and not _tile_aligned_pad and (bf16_shards or n_shards > 1):
            raise NotImplementedError(
                "EXL3 padded linear geometry is supported for single-shard layers "
                f"without bf16 shards only: in={true_in} out={true_out_sizes}"
            )
        if padded and not _tile_aligned_pad:
            _, _pad_tp = _resolve_tp_geometry(layer)
            if int(_pad_tp) > 1:
                raise NotImplementedError(
                    "EXL3 padded linear geometry requires tensor parallel size 1"
                )
            # vLLM registers the bias after create_weights with this instance's
            # weight_loader; a padded checkpoint bias is trimmed to the true size.
            _orig_loader = layer.weight_loader

            def _bias_trim_loader(param, loaded_weight, *args, **kwargs):
                if param.dim() == 1 and int(loaded_weight.shape[0]) > int(param.shape[0]):
                    loaded_weight = loaded_weight[: int(param.shape[0])]
                return _orig_loader(param, loaded_weight, *args, **kwargs)

            layer.weight_loader = _bias_trim_loader

        # Validate tile alignment for all shards
        for i, out_size in enumerate(output_partition_sizes):
            if in_per_partition % 16 or out_size % 16:
                raise ValueError(
                    f"EXL3 trellis tiles are 16-wide; "
                    f"shard {i}: in={in_per_partition} out={out_size}"
                )

        in_tiles = in_per_partition // 16
        out_tiles_list = [s // 16 for s in output_partition_sizes]
        total_out_tiles = sum(out_tiles_list)

        # Mixed-K merged groups: each shard gets its OWN trellis param
        # sized at its checkpoint K (the packed bitstream is K-dependent
        # per 16-value span — zero-padding a lower-K shard to a common
        # width corrupts dequantization). Uniform groups keep the fused
        # param (identical downstream behavior).
        if shard_ks and len(set(shard_ks)) > 1:
            trellis_params = [
                Parameter(
                    torch.empty(in_tiles, t, ks * 16, dtype=torch.int16),
                    requires_grad=False,
                )
                for t, ks in zip(out_tiles_list, shard_ks)
            ]
            trellis_param = trellis_params[0]
            layer._exl3_ragged_trellis = trellis_params
            layer._exl3_ragged_loaded = set()
        else:
            # Allocate fused trellis covering all shards (dim1 narrow per-shard)
            trellis_param = Parameter(
                torch.empty(in_tiles, total_out_tiles, k_words, dtype=torch.int16),
                requires_grad=False,
            )
        # Per-shard suh (one per shard, each covers this rank's input partition)
        # Batched (bmm) layers: each slice has its OWN suh (verified:
        # per-slice suh tensors are distinct) — allocate one row per
        # slice instead of one per shard.
        _suh_rows = bmm_slices if is_bmm and bmm_slices > n_shards else n_shards
        suh_param = Parameter(
            torch.empty(_suh_rows, in_per_partition, dtype=torch.float16),
            requires_grad=False,
        )
        # Per-shard svh (one per shard, concatenated)
        svh_param = Parameter(
            torch.empty(sum(output_partition_sizes), dtype=torch.float16),
            requires_grad=False,
        )
        # Per-shard mcg and mul1 markers (both registered, one will be nonzero)
        mcg_param = Parameter(
            torch.zeros(n_shards, 1, dtype=torch.int32),
            requires_grad=False,
        )
        mul1_param = Parameter(
            torch.zeros(n_shards, 1, dtype=torch.int32),
            requires_grad=False,
        )

        # Staging parameter for bf16 shards: rows are concatenated bf16 weights
        bf16_rows = sum(output_partition_sizes[i] for i in bf16_shards)
        weight_param = Parameter(
            torch.empty(bf16_rows, in_per_partition, dtype=params_dtype),
            requires_grad=False,
        )

        if padded:
            # Padded region never loaded — zero it so padded rows/cols
            # contribute nothing (padded svh = 0 → zero output).
            trellis_param.zero_()
            suh_param.zero_()
            svh_param.zero_()
        layer.register_parameter("trellis", trellis_param)
        layer.register_parameter("suh", suh_param)
        layer.register_parameter("svh", svh_param)
        layer.register_parameter("mcg", mcg_param)
        layer.register_parameter("mul1", mul1_param)
        layer.register_parameter("weight", weight_param)
        layer.register_parameter("trellis", trellis_param)

        # Custom weight loader
        extra = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}
        set_weight_attrs(trellis_param, extra)
        set_weight_attrs(suh_param, extra)
        set_weight_attrs(svh_param, extra)
        set_weight_attrs(mcg_param, extra)
        set_weight_attrs(mul1_param, extra)
        set_weight_attrs(weight_param, extra)

        # vLLM calls ``weight_loader(param, loaded_weight[, shard_id])`` and
        # never passes the checkpoint name, so bind the tensor kind per param.
        # Batched (bmm) layers: the checkpoint's slice tensors are
        # rank-local (the slice structure accounts for TP) — the
        # loader must NOT apply TP narrowing to them.
        for suffix, p in (
            ("trellis", trellis_param),
            ("suh", suh_param),
            ("svh", svh_param),
            ("mcg", mcg_param),
            ("mul1", mul1_param),
        ):
            p.weight_loader = self._make_weight_loader(
                suffix,
                n_shards,
                output_partition_sizes,
                is_row_parallel,
                bf16_shards,
                layer,
                is_qkv_parallel,
            )
        weight_param.weight_loader = self._make_weight_loader(
            "weight",
            n_shards,
            output_partition_sizes,
            is_row_parallel,
            bf16_shards,
            layer,
            is_qkv_parallel,
        )

        # Store metadata
        layer._exl3_linear_n_shards = n_shards
        layer._exl3_linear_output_partition_sizes = output_partition_sizes
        layer._exl3_linear_input_size_per_partition = in_per_partition
        layer._exl3_linear_is_row_parallel = is_row_parallel
        layer._exl3_linear_is_qkv = is_qkv_parallel
        layer._exl3_linear_is_merged = is_merged_col_parallel
        layer._exl3_linear_bf16_shards = bf16_shards
        layer._exl3_linear_padded = padded
        layer._exl3_linear_true_in = true_in
        layer._exl3_linear_true_out = true_out_sizes

    def _make_weight_loader(
        self,
        suffix,
        n_shards,
        output_partition_sizes,
        is_row_parallel,
        bf16_shards,
        layer=None,
        is_qkv_parallel=False,
        is_bmm=False,
        bmm_slices=0,
    ):
        """Create a weight_loader closure for EXL3 linear parameters."""

        def weight_loader(
            param: Parameter,
            loaded_weight: torch.Tensor,
            loaded_shard_id: str | int | None = None,
        ) -> None:
            tp_rank, tp_size = _resolve_tp_geometry(layer, param)

            # One checkpoint tensor may span several consecutive shards; vLLM's
            # WeightsMapper says so with a tuple of shard ids (Qwen3.5/4
            # in_proj_qkv -> in_proj_qkvz shards (0, 1, 2)). Split it along the
            # output dimension at the shard boundaries and load each piece.
            span_ids = None
            if isinstance(loaded_shard_id, (tuple, list)):
                span_ids = [int(i) for i in loaded_shard_id]
            elif loaded_shard_id is None and n_shards > 1:
                # Already-fused checkpoint tensor on a merged linear: an
                # output-sized tensor covering every shard is split; per-input
                # tensors and markers apply to every shard.
                if suffix in ("suh", "mcg", "mul1"):
                    span_ids = list(range(n_shards))
                else:
                    loaded_out = (
                        int(loaded_weight.shape[1]) * 16
                        if suffix == "trellis"
                        else int(loaded_weight.shape[0])
                    )
                    if loaded_out == sum(output_partition_sizes) and loaded_out != int(
                        output_partition_sizes[0]
                    ):
                        span_ids = list(range(n_shards))
            if span_ids is not None:
                ids = span_ids
                if (
                    not ids
                    or ids != list(range(ids[0], ids[0] + len(ids)))
                    or ids[-1] >= n_shards
                ):
                    raise ValueError(
                        f"EXL3 linear: unsupported shard id span {loaded_shard_id} "
                        f"for n_shards={n_shards}"
                    )
                if suffix == "suh":
                    # Suh rows are per-shard INPUT scales. The span
                    # tensor is one calibration group's suh shared by
                    # every covered shard (in_proj_qkv -> q,k,v: one
                    # [in_features] tensor) — write it FULL to each
                    # covered row. Slicing by output sizes corrupts:
                    # input-side scales have no per-shard boundaries.
                    # A concatenated layout (length == sum of row
                    # lengths) is sliced at row boundaries instead.
                    _row = (
                        int(param.data.shape[1])
                        if param.data.dim() == 2
                        else int(param.data.shape[0])
                    )
                    _len = int(loaded_weight.shape[0])
                    if _len == _row:
                        for i in ids:
                            weight_loader(param, loaded_weight, i)
                        return
                    if _len == _row * len(ids):
                        for _off, i in enumerate(ids):
                            weight_loader(
                                param,
                                loaded_weight[_off * _row : (_off + 1) * _row],
                                i,
                            )
                        return
                    start = 0
                    for i in ids:
                        size = output_partition_sizes[i]
                        weight_loader(param, loaded_weight[start : start + size], i)
                        start += size
                    return
                if suffix in ("mcg", "mul1"):
                    for i in ids:
                        weight_loader(param, loaded_weight, i)
                    return
                loaded_out = (
                    int(loaded_weight.shape[1]) * 16
                    if suffix == "trellis"
                    else int(loaded_weight.shape[0])
                )
                span = sum(output_partition_sizes[i] for i in ids)
                is_full_ckpt = False
                if loaded_out != span:
                    # TP-sharded merged layer: the checkpoint tensor covers
                    # the FULL output (every rank's span) while this rank's
                    # params hold only its shard. Slice pieces at FULL
                    # shard boundaries below — each recursive call narrows
                    # its piece to this rank's contiguous slice. Pre-narrow
                    # the whole tensor to this rank's contiguous slice
                    # instead and shards with unequal sizes interleave
                    # (v tiles landing on the q/k shards — audit-verified
                    # on the Qwen3.8-27B SC pack's in_proj_qkvz).
                    if loaded_out == span * tp_size:
                        is_full_ckpt = True
                    else:
                        raise RuntimeError(
                            f"EXL3 linear load: {suffix} tensor covers {loaded_out} outputs "
                            f"but shards {ids} total {span}"
                        )
                start = 0
                for i in ids:
                    size = output_partition_sizes[i] * (
                        tp_size if is_full_ckpt else 1
                    )
                    if suffix == "trellis":
                        piece = loaded_weight[:, start // 16 : (start + size) // 16, :]
                    else:
                        piece = loaded_weight[start : start + size]
                    weight_loader(param, piece, i)
                    start += size
                return

            # Map shard_id to shard index
            shard_idx = 0
            if loaded_shard_id is not None:
                if isinstance(loaded_shard_id, str):
                    # "q", "k", "v" for QKV layers
                    shard_map = {"q": 0, "k": 1, "v": 2}
                    if loaded_shard_id not in shard_map:
                        raise ValueError(
                            f"unknown shard_id={loaded_shard_id} for EXL3 linear"
                        )
                    shard_idx = shard_map[loaded_shard_id]
                elif isinstance(loaded_shard_id, int):
                    shard_idx = loaded_shard_id
            # Rank-local slice tensors (batched/bmm layers): the model
            # passes the LOCAL slice index; the param holds gpr
            # consecutive per-slice segments. Handle before the
            # n_shards range check (local idx can exceed n_shards=1).
            _ld0 = int(loaded_weight.shape[0]) if loaded_weight.dim() >= 1 else 0
            if (
                is_bmm
                and shard_idx < bmm_slices
                and suffix in ("svh", "suh", "trellis", "mul1", "mcg")
            ):
                _seg = _ld0  # per-slice segment length
                if suffix == "svh":
                    dest = param.data[shard_idx * _seg : (shard_idx + 1) * _seg]
                    if tuple(dest.shape) == tuple(loaded_weight.shape):
                        dest.copy_(loaded_weight)
                        return
                elif suffix == "suh":
                    dest = param.data[shard_idx]
                    if tuple(dest.shape) == tuple(loaded_weight.shape):
                        dest.copy_(loaded_weight)
                        return
                elif suffix in ("mul1", "mcg"):
                    # Markers: (n_shards, 1) param; both local slices'
                    # markers carry the same codebook value — write row 0.
                    if loaded_weight.numel() == 0:
                        return
                    param.data[0] = loaded_weight.reshape(1)
                    return
                else:  # trellis
                    _tiles = loaded_weight.shape[1]
                    _start = shard_idx * _tiles
                    dest = param.data[:, _start : _start + _tiles, :]
                    if tuple(dest.shape) == tuple(loaded_weight.shape):
                        dest.copy_(loaded_weight)
                        return
            _ragged = getattr(layer, "_exl3_ragged_trellis", None)
            if _ragged is not None and suffix == "trellis":
                # Ragged mixed-K layout: every shard's trellis lives in
                # _exl3_ragged_trellis[shard_idx] (the fused registered
                # param is only a name-routing vehicle and is deleted
                # post-load). Route THIS shard's write to its own param.
                # The write may arrive as a span (shard_id tuple covering
                # several shards, e.g. in_proj_qkv → (0,1,2)) — each
                # covered shard takes its out-tile slice at the actual
                # shard boundaries (uneven for GQA qkv).
                if isinstance(loaded_shard_id, (tuple, list)):
                    _qkv_map = {"q": 0, "k": 1, "v": 2}
                    ids = [
                        _qkv_map.get(i, i) if isinstance(i, str) else int(i)
                        for i in loaded_shard_id
                    ]
                    if shard_idx not in ids:
                        return
                    _pos = ids.index(shard_idx)
                    _span_tiles = [output_partition_sizes[i] // 16 for i in ids]
                    _lo = sum(_span_tiles[:_pos])
                    _hi = _lo + _span_tiles[_pos]
                    dest = _ragged[shard_idx].data
                    piece = loaded_weight[:, _lo : _hi, :]
                    # TP: the span piece covers tp_size ranks' worth of
                    # this shard's out-tiles; narrow to this rank's slice.
                    if piece.shape[1] == dest.shape[1] * tp_size and tp_size > 1:
                        _tlo = dest.shape[1] * tp_rank
                        piece = piece[:, _tlo : _tlo + dest.shape[1], :]
                    if tuple(piece.shape) != tuple(dest.shape):
                        raise RuntimeError(
                            f"EXL3 ragged trellis span load shape mismatch shard={shard_idx}: "
                            f"param {tuple(dest.shape)} != piece {tuple(piece.shape)}; "
                            f"prefix={getattr(layer, '_exl3_prefix', '?')}"
                        )
                    dest.copy_(piece)
                    layer._exl3_ragged_loaded.add(shard_idx)
                    return
                _qkv_map = {"q": 0, "k": 1, "v": 2}
                _sid = (
                    _qkv_map.get(loaded_shard_id, loaded_shard_id)
                    if isinstance(loaded_shard_id, str)
                    else loaded_shard_id
                )
                if _sid != shard_idx:
                    return
                dest = _ragged[shard_idx].data
                loaded = loaded_weight
                # TP: the checkpoint tensor covers tp_size ranks' worth
                # of out-tiles; narrow to this rank's contiguous slice.
                _rank_tiles = dest.shape[1]
                if loaded.shape[1] == _rank_tiles * tp_size and tp_size > 1:
                    _lo = _rank_tiles * tp_rank
                    loaded = loaded[:, _lo : _lo + _rank_tiles, :]
                if tuple(loaded.shape) != tuple(dest.shape):
                    raise RuntimeError(
                        f"EXL3 ragged trellis load shape mismatch shard={shard_idx}: "
                        f"param {tuple(dest.shape)} != loaded {tuple(loaded.shape)}; "
                        f"prefix={getattr(layer, '_exl3_prefix', '?')}"
                    )
                dest.copy_(loaded)
                layer._exl3_ragged_loaded.add(shard_idx)
                return
            if shard_idx >= n_shards:
                raise ValueError(
                    f"shard_idx={shard_idx} out of range for n_shards={n_shards}; "
                    f"suffix={suffix} loaded={tuple(loaded_weight.shape)} "
                    f"param={tuple(param.shape)} prefix={getattr(layer, '_exl3_prefix', '?')}"
                )

            # Special handling for weight (bf16 staging) and markers
            if suffix in ("weight", "mcg", "mul1"):
                # Weight parameter: only load bf16 shards, discard EXL3 shards
                if suffix == "weight":
                    # Check shape matches the expected shard size
                    expected_out = output_partition_sizes[shard_idx]
                    expected_in = param.shape[1]
                    loaded = loaded_weight.detach().contiguous()
                    loaded_shape = loaded.shape
                    if is_qkv_parallel and not is_row_parallel:
                        total_out = int(loaded_shape[0])
                        shard_tp_size = max(1, total_out // expected_out)
                        shard_tp_rank = tp_rank // max(1, tp_size // shard_tp_size)
                    else:
                        shard_tp_size = tp_size
                        shard_tp_rank = tp_rank
                    if tuple(loaded_shape) == (expected_out, expected_in):
                        tp_sharded = loaded
                    else:
                        # Row-parallel input is sharded on dim 1; column-parallel
                        # output is sharded on dim 0.
                        slice_dim = 1 if is_row_parallel else 0
                        tp_sharded = _narrow_tp(
                            loaded,
                            slice_dim,
                            shard_tp_rank,
                            shard_tp_size,
                        )
                    if tuple(tp_sharded.shape) != (expected_out, expected_in):
                        raise RuntimeError(
                            f"EXL3 weight load shape mismatch shard={shard_idx}: "
                            f"expected ({expected_out},{expected_in}) but got {tuple(loaded.shape)} "
                            f"(after TP: {tuple(tp_sharded.shape)}); "
                            f"prefix={getattr(layer, '_exl3_prefix', '?')}"
                        )
                    # If this shard is in bf16_shards, copy; otherwise discard
                    if shard_idx in bf16_shards:
                        bf16_idx = bf16_shards.index(shard_idx)
                        bf16_row_start = sum(output_partition_sizes[i] for i in bf16_shards[:bf16_idx])
                        bf16_row_end = bf16_row_start + expected_out
                        param.data[bf16_row_start:bf16_row_end].copy_(tp_sharded)
                    # else: discard this EXL3 shard's stale BF16 weight
                    return
                else:
                    # Marker (mcg or mul1): store the value (will be 0 if marker not present)
                    dest = param.data[shard_idx]
                    if tuple(dest.shape) != (1,):
                        raise RuntimeError(
                            f"EXL3 {suffix} marker shape mismatch: expected (1,) got {tuple(dest.shape)}"
                        )
                    loaded_val = loaded_weight.detach().item() if loaded_weight.numel() > 0 else 0
                    dest[0] = int(loaded_val)
                    return

            # Normal EXL3 suffix handling (trellis, suh, svh)
            loaded = loaded_weight.detach().contiguous()

            expected_out = output_partition_sizes[shard_idx]

            # Rank-local slice tensors (batched/bmm layers): when the
            # loaded tensor is smaller than the shard's expected out,
            # it is ONE slice's worth — already rank-local (the slice
            # structure accounts for TP). Write it at the slice's
            # segment within the shard: offset = shard_idx * loaded_len.
            total_out = (
                int(loaded.shape[1]) * 16
                if suffix == "trellis"
                else int(loaded.shape[0])
            )
            # round(): expected_out may be padded (tile-aligned pad);
            # the loaded tensor covers the unpadded geometry — integer
            # division would skew the shard count (129280//32384=3),
            # and the pad skews round() too (640 svh vs padded 256:
            # 640/256 = 2.5). The invariant: the loaded tensor covers
            # shard_tp_size ranks' worth of REAL rows, each piece must
            # fit the (possibly padded) dest, and shard_tp_size must
            # divide tp_size. Pick the COARSEST divisor whose per-rank
            # piece still fits — finer narrowing would under-fill the
            # dest's real rows; coarser pieces exceed the dest. For the
            # padded case (640 svh, padded 256 dest, tp 4) only d = 4
            # fits (640/4 = 160 <= 256); for exact cases the piece
            # equals expected_out at the true divisor.
            _cands = [
                d for d in range(1, tp_size + 1)
                if tp_size % d == 0 and total_out % d == 0
                and total_out // d <= expected_out
            ]
            shard_tp_size = next(
                iter(_cands), max(1, round(total_out / expected_out))
            )
            shard_tp_rank = tp_rank // max(1, tp_size // shard_tp_size)

            # Rank-local slice tensors (batched/bmm layers): when the
            # loaded tensor is smaller than the shard's expected out,
            # it is ONE slice's worth — already rank-local (the slice
            # structure accounts for TP). Write it at the slice's
            # segment within the shard: offset = shard_idx * loaded_len.
            _ld0 = int(loaded.shape[0]) if loaded.dim() >= 1 else 0
            # The output-side size of the loaded tensor: trellis dim 1
            # (out-tiles × 16); svh/suh dim 0. Comparing the trellis's
            # dim 0 (in-tiles) against expected_out is dimensionally
            # wrong — it misroutes merged-layer shard writes (the wkv
            # trellis landed inside the wq_a region).
            _out_side = (
                int(loaded.shape[1]) * 16
                if suffix == "trellis" and loaded.dim() >= 2
                else _ld0
            )
            if 0 < _out_side < expected_out and expected_out % _out_side == 0:
                if suffix == "svh":
                    dest = param.data[shard_idx * _ld0 : (shard_idx + 1) * _ld0]
                    if tuple(dest.shape) == tuple(loaded.shape):
                        dest.copy_(loaded)
                        return
                elif suffix == "suh":
                    dest = param.data[shard_idx]
                    if tuple(dest.shape) == tuple(loaded.shape):
                        dest.copy_(loaded)
                        return
                elif suffix == "trellis":
                    _tiles = loaded.shape[1]
                    _start = shard_idx * _tiles
                    dest = param.data[:, _start : _start + _tiles, :]
                    if tuple(dest.shape) == tuple(loaded.shape):
                        dest.copy_(loaded)
                        return

            # Vocab-parallel full tensor (quantized LM head): the
            # checkpoint ships the unsharded tensor; narrow to this
            # rank's contiguous range and write into the padded param
            # (the pad region was zeroed at allocation).
            # Gated on the LM head: without the gate this branch
            # catches EVERY tp_size=1 trellis call and routes merged
            # layers' shard-1 writes into the shard-0 region (the
            # wkv partition landed at [0:32] and was overwritten).
            _lp_name = str(getattr(layer, "_exl3_prefix", "") or getattr(layer, "prefix", ""))
            if (
                _lp_name.endswith("lm_head")
                and _ld0 > 0
                and not is_row_parallel
                and (
                    _ld0 >= param.shape[0] * (tp_size - 1)
                    and _ld0 % tp_size == 0
                    or (
                        suffix == "trellis"
                        and loaded.shape[1] >= param.shape[1] * (tp_size - 1)
                        and loaded.shape[1] % tp_size == 0
                    )
                )
            ):
                if suffix == "trellis":
                    _tiles = loaded.shape[1] // tp_size
                    dest = param.data[:, :_tiles, :]
                    _src = loaded[:, _tiles * tp_rank : _tiles * (tp_rank + 1), :]
                    if tuple(dest.shape) == tuple(_src.shape):
                        dest.copy_(_src)
                        return
                else:
                    _per = _ld0 // tp_size
                    dest = param.data[:_per]
                    _src = loaded[_per * tp_rank : _per * (tp_rank + 1)]
                    if tuple(dest.shape) == tuple(_src.shape):
                        dest.copy_(_src)
                        return

            # Apply TP slicing based on layer type
            if is_row_parallel:
                # Row-parallel: input is sharded, trellis dim 0 and suh dim 0
                sharded = shard_exl3_row(loaded, suffix, tp_rank, tp_size)
            else:
                # Column-parallel: output is sharded, trellis dim 1 and svh dim 0
                sharded = shard_exl3_col(
                    loaded, suffix, shard_tp_rank, shard_tp_size
                )

            # Copy into the right location
            if suffix == "trellis":
                # Trellis is fused; narrow dim1 for this shard
                out_tiles_start = sum(s // 16 for s in output_partition_sizes[:shard_idx])
                out_tiles_end = out_tiles_start + output_partition_sizes[shard_idx] // 16
                dest = param.data[:, out_tiles_start:out_tiles_end, :]
            elif suffix == "suh":
                # Suh per-shard
                dest = param.data[shard_idx]
            elif suffix == "svh":
                # Svh is concatenated; slice for this shard
                out_start = sum(output_partition_sizes[:shard_idx])
                out_end = out_start + output_partition_sizes[shard_idx]
                dest = param.data[out_start:out_end]
            else:
                raise ValueError(f"unknown EXL3 suffix={suffix}")

            if tuple(dest.shape) != tuple(sharded.shape):
                # Padded geometry: the narrowed tensor covers the shard's
                # TRUE input rows; the padded param row holds them at the
                # head with the pad region zeroed (padded suh rows are
                # zero — the pad contributes nothing).
                if (
                    suffix in ("suh", "svh")
                    and sharded.dim() == 1
                    and dest.shape[0] > sharded.shape[0]
                ):
                    dest[: sharded.shape[0]].copy_(sharded)
                    return
                if (
                    suffix == "trellis"
                    and sharded.dim() == 3
                    and dest.shape[0] > sharded.shape[0]
                    and dest.shape[1:] == sharded.shape[1:]
                ):
                    # Padded row-parallel input: the narrowed trellis holds
                    # the shard's TRUE in-tiles; the padded param's leading
                    # in-tile dim carries them at the head (pad tiles are
                    # zero — padded input rows only see zero-extended data).
                    dest[: sharded.shape[0]].copy_(sharded)
                    return
                if (
                    suffix == "trellis"
                    and sharded.dim() == 3
                    and dest.shape[0] == sharded.shape[0]
                    and dest.shape[1] > sharded.shape[1]
                    and dest.shape[2] == sharded.shape[2]
                ):
                    # Padded column-parallel output: the narrowed trellis
                    # holds the shard's TRUE out-tiles; the padded param's
                    # out-tile dim carries them at the head (pad tiles are
                    # zero — padded output rows contribute nothing).
                    dest[:, : sharded.shape[1]].copy_(sharded)
                    return
                # Rank-local checkpoint tensors (batched/bmm layers whose
                # slice structure already accounts for TP): the TP narrowing
                # above is wrong for them. Retry with the un-narrowed
                # tensor — if it fits the dest segment exactly, accept it.
                if tuple(dest.shape) == tuple(loaded.shape):
                    dest.copy_(loaded)
                    return
                raise RuntimeError(
                    f"EXL3 linear load shape mismatch shard={shard_idx} "
                    f"suffix={suffix}: dest {tuple(dest.shape)} != "
                    f"loaded {tuple(sharded.shape)}; "
                    f"prefix={getattr(layer, '_exl3_prefix', '?')} "
                    f"bits={getattr(self, 'bits', '?')} hb={getattr(self, 'head_bits', '?')}"
                )
            dest.copy_(sharded)
            if dest.device.type == "cuda" and torch is not None:
                try:
                    torch.cuda.current_stream().synchronize()
                except Exception:
                    pass
            _madv_dontneed_cpu_tensor(sharded)

            # Equal-shape case: dest was selected for exactly this shard, so
            # copy unconditionally (idempotent for double-writes). Without
            # this, replicated tensors whose shapes already match — e.g. the
            # lm_head's replicated suh — fell off the end of the closure and
            # silently left the param at its torch.empty allocation.
            dest.copy_(sharded)
            return
        return weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not hasattr(layer, "trellis"):
            return

        # Get bf16 shards and verify exactly one marker per EXL3 shard
        n_shards = int(layer._exl3_linear_n_shards)
        output_sizes = layer._exl3_linear_output_partition_sizes
        bf16_shards = getattr(layer, "_exl3_linear_bf16_shards", [])

        mcg_vals = layer.mcg.reshape(-1)
        mul1_vals = layer.mul1.reshape(-1)

        for i in range(n_shards):
            # Skip marker checks for bf16 shards - they don't use LinearEXL3
            if i in bf16_shards:
                continue
            mcg_is_set = mcg_vals[i].item() != 0
            mul1_is_set = mul1_vals[i].item() != 0
            if mcg_is_set and mul1_is_set:
                raise RuntimeError(
                    f"EXL3 linear shard {i}: both mcg and mul1 markers are set; "
                    f"exactly one codebook marker must be present"
                )
            if not mcg_is_set and not mul1_is_set:
                prefix = getattr(layer, 'prefix', '?')
                raise RuntimeError(
                    f"EXL3 linear {prefix} shard {i}: neither mcg nor mul1 marker is set; "
                    f"exactly one codebook marker must be present"
                )
            # Verify marker value
            if mcg_is_set and mcg_vals[i].item() != MCG_MARKER_SIGNED_INT32:
                raise RuntimeError(
                    f"EXL3 linear shard {i}: mcg marker is {mcg_vals[i].item()}, "
                    f"expected {MCG_MARKER_SIGNED_INT32}"
                )
            if mul1_is_set and mul1_vals[i].item() != MUL1_MARKER_SIGNED_INT32:
                raise RuntimeError(
                    f"EXL3 linear shard {i}: mul1 marker is {mul1_vals[i].item()}, "
                    f"expected {MUL1_MARKER_SIGNED_INT32}"
                )

        # Build LinearEXL3 objects for EXL3 shards only (skip bf16 shards)
        # Batched (bmm) layers: one linear per SLICE (the layer's single
        # vLLM partition covers bmm_slices group matrices). Slice i's
        # trellis occupies out-tile span [i*t : (i+1)*t] where t =
        # out_tiles // bmm_slices; suh row i; svh span [i*s : (i+1)*s]
        # with s = svh_numel // bmm_slices.
        _is_bmm = bool(getattr(layer, "_exl3_linear_is_bmm", False))
        _bmm_n = int(getattr(layer, "_exl3_bmm_slices", 0) or 0)
        _n_build = _bmm_n if _is_bmm and _bmm_n > 1 else n_shards
        linears = []
        for i in range(_n_build):
            if _is_bmm and _bmm_n > 1:
                _t = layer.trellis.shape[1] // _bmm_n
                _s = layer.svh.shape[0] // _bmm_n
                trellis_shard = layer.trellis[:, i * _t : (i + 1) * _t, :].contiguous()
                suh_shard = layer.suh[i].contiguous()
                svh_shard = layer.svh[i * _s : (i + 1) * _s].contiguous()
                # Markers are uniform across slices (loader comment 2970)
                mcg_shard = layer.mcg[0].contiguous() if layer.mcg[0].item() != 0 else None
                mul1_shard = layer.mul1[0].contiguous() if layer.mul1[0].item() != 0 else None
                linears.append(
                    make_linear_exl3(trellis_shard, suh_shard, svh_shard, mcg_shard, mul1_shard, out_dtype=torch.float16)
                )
                continue
            if i in bf16_shards:
                # bf16 shards don't use LinearEXL3; store None as placeholder
                linears.append(None)
                continue
            out_tiles_start = sum(s // 16 for s in output_sizes[:i])
            out_tiles_end = out_tiles_start + output_sizes[i] // 16
            _ragged = getattr(layer, "_exl3_ragged_trellis", None)
            if _ragged is not None:
                _loaded = getattr(layer, "_exl3_ragged_loaded", set())
                if _loaded != set(range(len(_ragged))):
                    _pfx = getattr(layer, "_exl3_prefix", "?")
                    raise RuntimeError(
                        f"EXL3 ragged trellis incomplete at {_pfx}: loaded "
                        f"shards {sorted(_loaded)} of {len(_ragged)} — mixed-K "
                        f"routing dropped a shard write"
                    )
                trellis_shard = _ragged[i].contiguous()
            else:
                trellis_shard = layer.trellis[:, out_tiles_start:out_tiles_end, :].contiguous()
            suh_shard = layer.suh[i].contiguous()
            svh_shard = layer.svh[
                sum(output_sizes[:i]) : sum(output_sizes[: i + 1])
            ].contiguous()
            mcg_shard = layer.mcg[i].contiguous() if mcg_vals[i].item() != 0 else None
            mul1_shard = layer.mul1[i].contiguous() if mul1_vals[i].item() != 0 else None

            linear = make_linear_exl3(
                trellis_shard, suh_shard, svh_shard, mcg_shard, mul1_shard, out_dtype=torch.float16
            )
            linears.append(linear)

        layer._exl3_linears = linears

        # Temp tensor audit (env-gated): dump per-shard checksums post-load.
        # VLLM_EXL3_TENSOR_AUDIT=<file> appends one JSON line per module.
        _audit_file = os.environ.get("VLLM_EXL3_TENSOR_AUDIT")
        if _audit_file:
            import hashlib
            from vllm.distributed import (
                get_tensor_model_parallel_rank as _gtr,
                get_tensor_model_parallel_world_size as _gtw,
            )

            def _audit_hash(t):
                t = t.detach().contiguous().cpu().reshape(-1)
                return [list(t.shape), hashlib.sha256(t.numpy().tobytes()).hexdigest()]

            _rec = {
                "prefix": getattr(layer, "_exl3_prefix", "?"),
                "rank": _gtr(),
                "tp": _gtw(),
                "shards": [],
            }
            _rag = getattr(layer, "_exl3_ragged_trellis", None)
            for _i in range(n_shards):
                if _i in bf16_shards:
                    _rec["shards"].append({"i": _i, "bf16": True})
                    continue
                _t_lo = sum(s // 16 for s in output_sizes[:_i])
                _t_hi = _t_lo + output_sizes[_i] // 16
                _t = (
                    _rag[_i]
                    if _rag is not None
                    else layer.trellis[:, _t_lo : _t_hi, :]
                )
                _s_lo = sum(output_sizes[:_i])
                _rec["shards"].append(
                    {
                        "i": _i,
                        "trellis": _audit_hash(_t),
                        "suh": _audit_hash(layer.suh[_i]),
                        "svh": _audit_hash(layer.svh[_s_lo : _s_lo + output_sizes[_i]]),
                        "mul1": int(layer.mul1[_i].item()),
                        "mcg": int(layer.mcg[_i].item()),
                    }
                )
            with open(_audit_file, "a") as _af:
                _af.write(json.dumps(_rec) + "\n")

        layer._exl3_opaque_name = _exl3_register_opaque_layer(layer, "linear")

        # Keep bf16 weights if present, remove weight staging param if all loaded
        if bf16_shards and hasattr(layer, "weight"):
            bf16_rows = sum(output_sizes[i] for i in bf16_shards)
            if bf16_rows > 0:
                layer._exl3_bf16_weight = layer.weight.data.clone()
            # Delete weight staging param only if it has rows; empty param stays as placeholder
            if layer.weight.data.shape[0] > 0:
                try:
                    delattr(layer, "weight")
                except Exception:
                    pass
        elif hasattr(layer, "weight"):
            # No bf16 shards, delete the staging param
            try:
                delattr(layer, "weight")
            except Exception:
                pass

        # Free fused parameters to avoid memory doubling
        for param_name in ("trellis", "suh", "svh", "mcg", "mul1"):
            if hasattr(layer, param_name):
                try:
                    delattr(layer, param_name)
                except Exception:
                    pass

    def apply(
        self,
        layer,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        name = getattr(layer, "_exl3_opaque_name", None)
        if name is not None and _EXL3_OPS_READY:
            y = torch.ops.vllm.exl3_linear_forward(x, name)
        else:
            y = self._apply_impl(layer, x)
        if bias is not None:
            y = y + bias
        return y

    def _apply_impl(self, layer, x: torch.Tensor) -> torch.Tensor:
        linears = getattr(layer, "_exl3_linears", None)
        if _EXL3_PREFILL_SYNC:
            _prefill_sync(int(x.numel() // x.shape[-1]))
        if not linears:
            raise RuntimeError("EXL3 linear layers were not built after weight load")

        # x shape: (batch, in_features) or (batch, ..., in_features)
        # Flatten to 2D: (rows, in_features)
        orig_shape = x.shape
        if len(orig_shape) > 2:
            # Multi-dim input: flatten to (rows, in)
            rows = 1
            for d in orig_shape[:-1]:
                rows *= d
            x_2d = x.reshape(rows, orig_shape[-1])
        else:
            x_2d = x

        # Batched (bmm) layers: slice i pairs with group i along the
        # group dim (dim -2 of the 3D input). Each slice processes ONLY
        # its own group's rows; outputs concatenate along the last dim.
        # (Running all slices on all rows and concatenating would give
        # each group every slice's output — doubling z's width.)
        _bmm_slices = getattr(layer, "_exl3_bmm_slices", 0)
        if (
            _bmm_slices > 1
            and len(orig_shape) == 3
            and orig_shape[-2] == _bmm_slices
        ):
            outs = []
            for i in range(_bmm_slices):
                linear = linears[i]
                if linear is None:
                    raise RuntimeError(f"EXL3 bmm slice {i} is None")
                xi = x[:, i, :].to(torch.float16).contiguous()
                outs.append(_dense_forward(linear, xi))
            y = torch.cat(outs, dim=-1).to(dtype=x.dtype)
            return y

        # Cast to contiguous fp16 for EXL3 shards
        x_fp16 = x_2d.to(torch.float16).contiguous()
        if getattr(layer, "_exl3_linear_padded", False):
            pad_in = int(layer._exl3_linear_input_size_per_partition) - int(x_fp16.shape[1])
            if pad_in > 0:
                x_fp16 = F.pad(x_fp16, (0, pad_in))

        # Get bf16 shards and weight if present
        bf16_shards = getattr(layer, "_exl3_linear_bf16_shards", [])
        bf16_weight = getattr(layer, "_exl3_bf16_weight", None)
        output_sizes = layer._exl3_linear_output_partition_sizes
        n_shards = len(linears)

        # Run each shard in declared order
        outputs = []
        for i in range(n_shards):
            if i in bf16_shards:
                # BF16 shard: use dense linear
                if bf16_weight is None:
                    raise RuntimeError(
                        f"EXL3 bf16 shard {i} but _exl3_bf16_weight is missing"
                    )
                bf16_idx = bf16_shards.index(i)
                out_start = sum(output_sizes[j] for j in bf16_shards[:bf16_idx])
                out_end = out_start + output_sizes[i]
                w_shard = bf16_weight[out_start:out_end]
                out = F.linear(x_2d, w_shard).to(dtype=torch.float32)
                outputs.append(out)
            else:
                # EXL3 shard
                linear = linears[i]
                if linear is None:
                    raise RuntimeError(f"EXL3 linear shard {i} is None")
                out = _dense_forward(linear, x_fp16)
                outputs.append(out)

        # Concatenate shards along output dimension
        if len(outputs) > 1:
            y = torch.cat(outputs, dim=1)
        else:
            y = outputs[0]

        # Trim padded output columns (svh = 0 there, so they are zeros)
        if getattr(layer, "_exl3_linear_padded", False):
            y = y[:, : sum(layer._exl3_linear_true_out)]

        # Cast back to input dtype
        y = y.to(dtype=x.dtype)

        # Restore original shape
        if len(orig_shape) > 2:
            y = y.reshape(*orig_shape[:-1], y.shape[-1])

        return y
