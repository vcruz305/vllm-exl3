"""ATS zero-copy weight aliases (DGX Spark / GB10).

On a GB10 the GPU runs in ATS addressing mode: it shares the process page
tables, so a device pointer is just a host virtual address.  A read-only
``mmap`` of a file can therefore be handed to the kernels directly, with no
CUDA allocation and no copy: the weights live in reclaimable page cache and the
GPU faults them in from NVMe as it touches them.

This module is the vLLM-side equivalent of the owner's
``EXL3_ATS_MMAP`` loader path (``exllamav3/loader/safetensors.py`` +
``exllamav3_ext.ats_cuda_view``): the same ``ats_cuda_view`` primitive from
the installed exllamav3 fork, applied to

* ``embed.weight`` -- aliased straight out of the pack's shard (bf16, 16-byte
  aligned in the v2 pack), and
* whole routed-expert trellis tensors -- aliased out of a 64-byte-aligned
  side file built once by ``tools/ats_build_alias.py`` (the pack itself puts
  every expert trellis on a 12-mod-16 offset, which the int16 trellis kernels
  reject).

Everything is behind env switches and defaults to off:

``VLLM_EXL3_ATS_ALIAS=1``
    master switch (anything else: no-op, byte-identical to the normal path).
``VLLM_EXL3_ATS_ALIAS_DIR``
    directory holding ``manifest.json`` + the aligned side file(s).
``VLLM_EXL3_ATS_ALIAS_LAYERS``
    comma list of routed-expert layer ids to alias, e.g. ``"38,39"``.
``VLLM_EXL3_ATS_ALIAS_MTP``
    ``1``: alias the ``mtp.N.ffn.experts`` (DSpark) layers listed in
    ``VLLM_EXL3_ATS_ALIAS_LAYERS`` instead of/in addition to ``layers.N``;
    the manifest carries one entry per checkpoint prefix.
``VLLM_EXL3_ATS_ALIAS_EMBED=1``
    alias ``embed.weight`` from ``VLLM_EXL3_ATS_ALIAS_EMBED_FILE`` (default:
    resolved from the pack index next to ``VLLM_EXL3_MODEL_DIR``).
``VLLM_EXL3_ATS_ALIAS_MIN_BYTES``
    skip tensors smaller than this (default 1 MiB).
``VLLM_EXL3_ATS_PREFETCH=1``
    ``posix_fadvise(WILLNEED)`` the aliased ranges once at startup.
``VLLM_EXL3_ATS_PREFETCH_MAX_GIB``
    skip the startup prefetch when the side file is larger than this
    (default 0 = no limit).  Prefetching a file that does not fit in page
    cache evicts everything and guarantees the first steps miss; set this to
    ~0.7 x MemAvailable for a cold-set build.

Per-expert selection
--------------------
A manifest may carry ``"selection": "expert_list"`` and
``"experts": {"layers.38": [12, 40, ...], "mtp.1": [...]}``.  When it does,
the manifest is the sole authority on what is aliased and
``VLLM_EXL3_ATS_ALIAS_LAYERS`` / ``..._MTP`` are ignored: a layer can alias a
strict subset of its experts, which is what a routing-frequency cold set is.
An old manifest without ``experts`` keeps the layer-granular behaviour
unchanged.
"""

from __future__ import annotations

import json
import logging
import mmap
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Checkpoint expert-directory families the alias side file may address.
# Historical layout is layers.N.ffn.experts; MiniMax-M3 stores its routed
# experts under layers.N.block_sparse_moe.experts.
EXPERT_CKPT_PREFIXES = (
    "ffn.experts",
    "block_sparse_moe.experts",
)

# MiniMax-M3's embedding tensor name in the VL wrapper's checkpoint; the
# historical default stays "embed.weight".
MINIMAX_EMBED_KEY = "language_model.model.embed_tokens.weight"

_STATS: dict[str, Any] = {
    "aliased_bytes": 0,
    "aliased_tensors": 0,
    "skipped_fill_bytes": 0,
    "skipped_fill_calls": 0,
    "embed_bytes": 0,
    "files": [],
}

_MAPS: dict[str, Any] = {}
_MANIFEST: dict[str, Any] | None = None
_MANIFEST_LOADED = False
_LAYER_CACHE: dict[str, dict[str, dict[int, tuple]]] = {}


def _on() -> bool:
    return os.environ.get("VLLM_EXL3_ATS_ALIAS", "0") not in ("0", "", "false", "False")


def _min_bytes() -> int:
    try:
        return int(os.environ.get("VLLM_EXL3_ATS_ALIAS_MIN_BYTES", str(1 << 20)))
    except ValueError:
        return 1 << 20


def _alias_layers() -> set[str]:
    raw = os.environ.get("VLLM_EXL3_ATS_ALIAS_LAYERS", "").strip()
    return {p.strip() for p in raw.split(",") if p.strip()}


def _manifest() -> dict[str, Any]:
    global _MANIFEST, _MANIFEST_LOADED
    if _MANIFEST_LOADED:
        return _MANIFEST or {}
    _MANIFEST_LOADED = True
    d = os.environ.get("VLLM_EXL3_ATS_ALIAS_DIR", "")
    if not d:
        return {}
    path = os.path.join(d, "manifest.json")
    try:
        with open(path, encoding="utf-8") as fh:
            man = json.load(fh)
    except Exception as exc:  # pragma: no cover - config error is fatal below
        logger.error("ATS alias: cannot read manifest %s: %r", path, exc)
        return {}
    man["_dir"] = d
    _MANIFEST = man
    return man


def _map_file(path: str):
    """Shared read-only mapping of one alias file, plus a uint8 view on it."""
    import torch

    if not os.path.isabs(path):
        path = os.path.join(_manifest()["_dir"], path)
    ent = _MAPS.get(path)
    if ent is not None:
        return ent
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
    if os.environ.get("VLLM_EXL3_ATS_HUGEPAGE", "0") not in ("0", ""):
        try:
            mm.madvise(mmap.MADV_HUGEPAGE)
        except Exception:
            pass
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        base = torch.frombuffer(mm, dtype=torch.uint8)  # keeps mmap alive
    ent = (mm, base)
    _MAPS[path] = ent
    _STATS["files"].append(path)
    return ent


def _ats_view(path: str, offset: int, nbytes: int, shape, dtype, device):
    """CUDA tensor aliasing ``[offset, offset+nbytes)`` of ``path`` (ATS)."""
    import torch

    if offset % 16:
        raise RuntimeError(f"ATS alias offset {offset} not 16-byte aligned: {path}")
    _mm, base = _map_file(path)
    # element-aligned slice, then reinterpret through the loader's own
    # region-coercion helper (the same reinterpret every checkpoint-shaped
    # tensor goes through).
    from .exl3 import coerce_checkpoint_region, load_exllamav3_ext

    itemsize = int(torch.tensor([], dtype=dtype).element_size())
    if offset % itemsize:
        raise RuntimeError(f"ATS alias offset {offset} not {itemsize}-byte aligned: {path}")
    t = coerce_checkpoint_region(
        base[offset : offset + nbytes], dtype, shape
    )
    ext = load_exllamav3_ext()
    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    out = ext.ats_cuda_view(t.contiguous(), idx)
    _STATS["aliased_bytes"] += int(nbytes)
    _STATS["aliased_tensors"] += 1
    return out


# ── routed experts ──────────────────────────────────────────────────────────


def _layer_key(layer: Any) -> str | None:
    name = str(
        getattr(layer, "layer_name", None)
        or getattr(layer, "prefix", None)
        or ""
    )
    m = re.search(r"(?:^|\.)((?:layers|mtp)\.(\d+))(?:\.|$)", name)
    if not m:
        return None
    return m.group(1)


def layer_aliases(layer: Any) -> dict[str, dict[int, tuple]] | None:
    """``{proj: {eid: (path, off, nbytes, shape)}}`` for this layer, or None."""
    if not _on():
        return None
    key = _layer_key(layer)
    if key is None:
        return None
    man = _manifest()
    tensors = man.get("tensors", {})
    # DSpark's draft layers are built as ``model.layers.<num_hidden_layers + i>``
    # while the checkpoint (and this manifest) names them ``mtp.<i>``. Translate
    # so a drafter layer resolves to the same side-file entries.
    if not key.startswith("mtp."):
        _off = man.get("mtp_layer_offset")
        _m = re.match(r"layers\.(\d+)$", key)
        if isinstance(_off, int) and _m and int(_m.group(1)) >= _off:
            _cand = f"mtp.{int(_m.group(1)) - _off}"
            if any(k.startswith(_cand + ".ffn.experts.") for k in tensors):
                key = _cand
    if key in _LAYER_CACHE:
        return _LAYER_CACHE[key] or None
    want = _alias_layers()
    short = key.split(".")[-1]
    is_mtp = key.startswith("mtp.")
    sel = man.get("experts")
    if sel is not None:
        # Expert-list build: the manifest decides, layer by layer, which expert
        # ids are aliased.  ``VLLM_EXL3_ATS_ALIAS_LAYERS`` is not consulted --
        # a cold-set manifest typically spans many layers with a handful of
        # experts each.
        eids: set[int] | None = {int(x) for x in (sel.get(key) or [])}
        if not eids:
            _LAYER_CACHE[key] = {}
            return None
    else:
        eids = None
        if is_mtp:
            # MTP/draft layers are selected by the side file itself
            # (`--mtp 0,1,2` at build time), not by the main-stack layer list.
            wanted = os.environ.get("VLLM_EXL3_ATS_ALIAS_MTP", "0") not in ("0", "", "false") and any(
                k.startswith(key + ".ffn.experts.") for k in tensors
            )
        else:
            wanted = short in want
        if not wanted:
            _LAYER_CACHE[key] = {}
            return None
    man = _manifest()
    tensors = man.get("tensors", {})
    out: dict[str, dict[int, tuple]] = {"gate": {}, "up": {}, "down": {}}
    proj_of = {"w1": "gate", "w3": "up", "w2": "down"}
    pat = re.compile(
        r"(?:^|\.)" + re.escape(key) + r"\.("
        + "|".join(re.escape(p) for p in EXPERT_CKPT_PREFIXES)
        + r")\.(\d+)\.(w1|w2|w3)\.trellis$"
    )
    for k, ent in tensors.items():
        m = pat.search(k)
        if not m:
            continue
        eid = int(m.group(1))
        if eids is not None and eid not in eids:
            continue
        proj = proj_of[m.group(2)]
        off, nbytes = int(ent[0]), int(ent[1])
        shape = tuple(int(x) for x in ent[3])
        out[proj][eid] = (man.get("file", "experts.bin"), off, nbytes, shape)
    if not any(out.values()):
        _LAYER_CACHE[key] = {}
        return None
    _LAYER_CACHE[key] = out
    return out


def is_aliased(layer: Any, proj: str, expert_id: int) -> bool:
    aliases = layer_aliases(layer)
    return bool(aliases and expert_id in aliases.get(proj, {}))


def alias_shape(layer: Any, proj: str, expert_id: int):
    aliases = layer_aliases(layer) or {}
    ent = aliases.get(proj, {}).get(expert_id)
    return ent[3] if ent else None


def install_layer_aliases(layer: Any, device) -> int:
    """Materialize ATS views for this layer's aliased experts into ``plist``.

    Returns the number of experts aliased (0 when the layer is not in the
    alias set).  Called from ``prepare_trellis_arena_plan`` before the arenas
    are sized, so aliased experts never reserve CUDA bytes.
    """
    aliases = layer_aliases(layer)
    if not aliases:
        return 0
    import torch
    from torch.nn import Parameter

    proj_to_plist = {
        "gate": layer.gate_trellis,
        "up": layer.up_trellis,
        "down": layer.down_trellis,
    }
    n = 0
    for proj, eids in aliases.items():
        plist = proj_to_plist.get(proj)
        if plist is None:
            continue
        for eid, (path, off, nbytes, shape) in sorted(eids.items()):
            view = _ats_view(path, off, nbytes, shape, torch.int16, device)
            p = Parameter(view, requires_grad=False)
            p.weight_loader = getattr(plist[eid], "weight_loader", None)
            p._exl3_owner = layer
            p._exl3_ats_aliased = True
            plist[eid] = p
            n += 1
    if n:
        logger.info(
            "EXL3 ATS alias: layer=%s experts aliased=%d bytes=%.2f GiB",
            _layer_key(layer),
            n,
            sum(e[2] for m in aliases.values() for e in m.values()) / 2**30,
        )
    return n


def maybe_alias_expert(layer: Any, proj: str, expert_id: int, device):
    """Parameter aliasing one expert trellis, or None (legacy per-expert path).

    Used by layers with no arena plan (the ``mtp.*`` DSpark layers): the loader
    builds one Parameter per expert, so the alias simply replaces that
    allocation.
    """
    aliases = layer_aliases(layer) or {}
    ent = aliases.get(proj, {}).get(int(expert_id))
    if ent is None:
        return None
    import torch
    from torch.nn import Parameter

    path, off, nbytes, shape = ent
    view = _ats_view(path, off, nbytes, shape, torch.int16, device)
    p = Parameter(view, requires_grad=False)
    p._exl3_owner = layer
    p._exl3_ats_aliased = True
    _STATS["skipped_fill_calls"] += 1
    _STATS["skipped_fill_bytes"] += int(nbytes)
    return p


# ── embed ──────────────────────────────────────────────────────────────────


def _pack_index() -> dict[str, str]:
    d = os.environ.get("VLLM_EXL3_MODEL_DIR") or os.environ.get("VLLM_ENGRAM_MODEL_DIR")
    if not d:
        return {}
    try:
        with open(os.path.join(d, "model.safetensors.index.json"), encoding="utf-8") as fh:
            return json.load(fh).get("weight_map", {})
    except Exception:
        return {}


def _tensor_entry(model_dir: str, shard: str, key: str):
    path = os.path.join(model_dir, shard)
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        hdr = json.loads(fh.read(n))
    b, e = hdr[key]["data_offsets"]
    return path, 8 + n + b, e - b, tuple(hdr[key]["shape"])


def alias_embedding(model: Any) -> int:
    """Replace the text embedding weight with an ATS view of the pack shard."""
    if not _on():
        return 0
    if os.environ.get("VLLM_EXL3_ATS_ALIAS_EMBED", "1") in ("0", "", "false"):
        return 0
    import torch
    from torch.nn import Parameter

    model_dir = os.environ.get("VLLM_EXL3_MODEL_DIR") or os.environ.get(
        "VLLM_ENGRAM_MODEL_DIR", ""
    )
    idx = _pack_index()
    key = os.environ.get("VLLM_EXL3_ATS_ALIAS_EMBED_KEY", "embed.weight")
    shard = idx.get(key)
    if shard is None and "VLLM_EXL3_ATS_ALIAS_EMBED_KEY" not in os.environ:
        # MiniMax-M3 VL wrapper: the text embedding is stored under its
        # language_model.model.* name, not the historical embed.weight.
        if idx.get(MINIMAX_EMBED_KEY) is not None:
            key = MINIMAX_EMBED_KEY
            shard = idx[key]
    # Prefer the 64-byte-aligned side-file copy when it was built with
    # ``--embed``: v3 puts ``embed.weight`` at an 8-mod-16 offset in its shard and
    # the ATS view needs 16, so the side file is the only usable source there.
    path = off = nbytes = shape = None
    man = _manifest()
    ent = (man.get("tensors") or {}).get(man.get("embed_key") or "") if man else None
    if ent is not None:
        off, nbytes, _dt, sh = ent
        path = os.path.join(man.get("_dir") or "", man.get("file", "experts.bin"))
        shape = tuple(int(x) for x in sh)
    else:
        if not (model_dir and shard):
            logger.warning("ATS alias: embed.weight not found in %s index", model_dir)
            return 0
        path, off, nbytes, shape = _tensor_entry(model_dir, shard, key)
    done = 0
    for name, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if not torch.is_tensor(w) or w.device.type != "cuda":
            continue
        if tuple(w.shape) != shape:
            continue
        if "embed_tokens" not in name and not name.endswith("embed"):
            continue
        view = _ats_view(path, off, nbytes, shape, w.dtype, w.device)
        p = Parameter(view, requires_grad=False)
        for attr in ("weight_loader", "_exl3_owner"):
            if hasattr(w, attr):
                try:
                    setattr(p, attr, getattr(w, attr))
                except Exception:
                    pass
        mod.weight = p
        _STATS["embed_bytes"] += int(nbytes)
        done += 1
        logger.info(
            "EXL3 ATS alias: %s -> %s[%d:+%d] %.2f GiB (off=%d, %d%%64=%d)",
            name,
            os.path.basename(path),
            off,
            nbytes,
            nbytes / 2**30,
            off,
            off % 64,
        )
    return done


def prefetch() -> int:
    """``posix_fadvise(WILLNEED)`` every aliased range once."""
    if not _on():
        return 0
    if os.environ.get("VLLM_EXL3_ATS_PREFETCH", "1") in ("0", "", "false"):
        return 0
    man = _manifest()
    if not man:
        return 0
    f = man.get("file", "experts.bin")
    path = os.path.join(man["_dir"], f)
    if not os.path.isfile(path):
        return 0
    try:
        max_gib = float(os.environ.get("VLLM_EXL3_ATS_PREFETCH_MAX_GIB", "0") or 0)
    except ValueError:
        max_gib = 0.0
    if max_gib > 0:
        size_gib = os.path.getsize(path) / 2**30
        if size_gib > max_gib:
            logger.info(
                "EXL3 ATS alias: skipping prefetch of %.2f GiB (limit %.2f GiB) "
                "- it would evict page cache it cannot refill",
                size_gib,
                max_gib,
            )
            return 0
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
    finally:
        os.close(fd)
    return 1


def stats() -> dict[str, Any]:
    out = dict(_STATS)
    out["aliased_gib"] = round(_STATS["aliased_bytes"] / 2**30, 3)
    out["skipped_fill_gib"] = round(_STATS["skipped_fill_bytes"] / 2**30, 3)
    return out


def log_stats() -> None:
    s = stats()
    logger.info(
        "EXL3 ATS alias: tensors=%d aliased=%.2f GiB embed=%.2f GiB "
        "fill_skipped=%.2f GiB/%d calls",
        s["aliased_tensors"],
        s["aliased_gib"],
        s["embed_bytes"] / 2**30,
        s["skipped_fill_gib"],
        s["skipped_fill_calls"],
    )