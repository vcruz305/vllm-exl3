"""Import-time overlay: 4-bit (nvfp4-layout) main MSA KV for MiniMax-M3.

Installed by ``vllm_exl3.__init__`` only when ``VLLM_EXL3_Q4KV=1``. Default
off: nothing is patched and behavior is byte-identical.

Milestone 1 scope (main MSA KV only; the indexer cache and the DSpark
draft pool stay fp8 and are untouched):

- ``select_main_backend_and_impl_cls`` is rerouted so a ``nvfp4`` main KV
  on non-SM100 CUDA (GB10/sm_121) selects this overlay's backend, which
  publishes the upstream nvfp4 spec arithmetic (2 K/V slots per head,
  72 B per slot, LBHNC) so the pool is sized at 576 B/token/layer for
  4 KV heads. The implementation class stays the vendor Triton impl; its
  read/write entry points are rerouted below.
- ``vllm._custom_ops.fused_minimax_m3_qknorm_rope_kv_insert`` is wrapped:
  a q4 pool is served by our Triton W1 (Gemma QK-norm + partial NeoX RoPE
  + e2m1/e4m3 pack + scatter insert); every other pool (fp8/bf16, and the
  DSpark draft, which uses a separate op) goes to the original CUDA op
  unchanged.
- ``minimax_m3_sparse_attn_decode`` is wrapped: a q4 pool is served by our
  Triton R2 (q4-dequant split-K decode + the unchanged vLLM merge kernel).
- ``minimax_m3_sparse_attn`` (prefill) is wrapped: a q4 pool is served by
  the milestone-1 staging path -- the selected physical pages are
  dequantized to bf16 and the original prefill kernel runs on the staged
  pool. Correct but slow; a fused q4 prefill reader (R1) is milestone 2.
  The staged union is memory-bounded: one grow-once bf16 buffer per
  device, filled in page-capped slices
  (``VLLM_EXL3_Q4KV_STAGING_MAX_MB``, default 48 MiB), and the sliced
  result equals the uncapped staging bit-for-bit (pinned by tests and the
  harness dry run against the fp32 prefill oracle).
  ``VLLM_EXL3_Q4KV_PREFILL_FAST=1`` (default off) replaces both halves of
  that path with single-pass equivalents -- one Triton launch per slice
  over the q4 pool instead of one launch group per 2 pages through an
  int32/float32 chain, and O(1) host syncs per slice instead of one per
  row. Same slice plan, same staged bytes, ~8x less traffic; see
  ``lanes/research/q4-prefill-notes.md`` for the cost model.

All wrappers fail closed: they raise on any unsupported combination rather
than silently miscomputing.

CUDA-graph capture safety (the boot20 fix and audit): vLLM serves decode
under FULL_DECODE_ONLY CUDA graphs, which capture DECODE steps only --
``CUDAGraphMode.FULL_DECODE_ONLY = (FULL, NONE)`` maps decode batches to
FULL and every mixed/prefill batch to NONE/eager
(vllm/config/compilation.py:62,65-67,96), and the capture set is built
from decode descriptors only (vllm/v1/worker/gpu/cudagraph_utils.py:
264-266,302-343). Every captured forward is first run EAGERLY as a
warm-up pass -- ``create_forward_fn(desc, warmup=True)`` then
``forward_fn(CUDAGraphMode.NONE)`` at cudagraph_utils.py:464-467, before
``torch.cuda.graph`` at :498-503 -- so anything the captured step touches
is exercised eagerly first. The q4 hooks are therefore capture-safe by
construction:

- No hook reads a tensor VALUE on the decode path. Pool detection
  (``_is_q4_pool``), the byte re-view (``_q4_pool_bytes``), the device
  gates and every geometry check inspect metadata only
  (device/dtype/shape/stride); the Triton launches take their grids and
  scalar arguments from shapes and Python scalars.
- The unit-scale gate (``_assert_unit_scale``) validates each scale
  tensor ONCE eagerly and caches the verdict against the exact tensor
  object; the same object seen inside a captured step is approved with
  no read, and a scale first seen while capturing fails closed (see the
  function docstring for the vLLM warm-up evidence).
- Every scalar argument that a wrapper would coerce with
  ``bool``/``int``/``float`` is gated by ``_py_scalar``: a tensor in one
  of those slots is refused (coercion is a host sync) instead of
  converted.
- The prefill wrapper's ``.tolist()`` slice metadata
  (``cu_seqlens_q``/``prefix_lens``) is never captured: prefill batches
  run under the FULL_DECODE_ONLY mixed mode NONE (eager), per the mode
  mapping cited above.

Binding audit (why ``install`` patches module ATTRIBUTES on every importer):
upstream modules import the hooks BY VALUE (``from ... import hook``), so
patching only the defining module's attribute never reaches an importer that
was already loaded -- exactly the boot17 failure, where
``models/minimax_m3/nvidia/model.py:128``'s by-value import of
``select_main_backend_and_impl_cls`` kept the original SM100-only selector
and the engine died at model init. ``install`` therefore imports every
binding module itself and patches the attribute on each of them; the
binding sites are enumerated in ``_REQUIRED_BINDING_SITES`` /
``_OPTIONAL_BINDING_SITES`` and verified by a fail-closed self-check.
"""

from __future__ import annotations

import logging
import os
import weakref
from dataclasses import replace
from typing import Any

import torch

from vllm_exl3 import minimax_q4kv_reference as ref

logger = logging.getLogger(__name__)

_SPARSE_BLOCK_SIZE = 128


def _is_q4_pool(t: object) -> bool:
    """A pool container is ours iff it is the bound nvfp4 container
    [pages, 2*num_kv_heads, 128, 72] with contiguous slot regions. The fp8
    and bf16 pools ([.., H, 128, 256]) and the 3-D indexer pool never match.
    """
    return (
        isinstance(t, torch.Tensor)
        and t.dim() == 4
        and t.shape[2] == _SPARSE_BLOCK_SIZE
        and t.shape[3] == ref.ROW_BYTES
        and t.shape[1] % 2 == 0
        and t.stride(3) == 1
        and t.stride(2) == ref.ROW_BYTES
        and t.stride(1) == ref.SLOT_REGION
    )


def _stream_is_capturing() -> bool:
    """True iff the current CUDA stream is capturing a CUDA graph.

    The probe is inert (False) wherever the CUDA API is unavailable --
    e.g. the CPU-only test suite, where torch raises on the call: there
    is no stream to capture there, so nothing is ever "capturing".
    """
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001 - probe only, never a data path
        return False


# Verdicts of the unit-scale gate below: (device, data_ptr, version
# counter) -> weak reference to the EXACT tensor object whose value was
# checked. Keying on the address alone would be unsound (the allocator
# reuses freed addresses); the weakref binds approval to the live object,
# and a dead weakref is never an approval.
_VALIDATED_SCALES: dict = {}


def _assert_unit_scale(name: str, scale: object) -> None:
    """Unit-global-scale gate, capture-safe (the boot20 fix).

    Milestone 1 serves the unit global scale only. Reading the scale's
    VALUE (``.item()``) is a host sync, and CUDA forbids host syncs while
    a stream is capturing -- the real 32K serve died exactly there
    (``float(scale.reshape(-1)[0].item())`` inside CUDA-graph capture,
    minimax_q4kv.py's W1 wrapper -> cudagraph_utils.py:502). The verdict
    is therefore validated ONCE, eagerly, and cached against the exact
    tensor object; the same object seen again -- including inside a
    captured decode step -- is approved without any read. vLLM always
    runs every captured forward eagerly first (the warm-up pass:
    vllm/v1/worker/gpu/cudagraph_utils.py:464-467 invokes
    ``create_forward_fn(desc, warmup=True)`` and runs it under
    CUDAGraphMode.NONE BEFORE ``torch.cuda.graph`` at :498-503), and the
    production scale tensors are persistent per-layer attributes, so the
    eager warm-up always validates them before any capture. A scale
    FIRST seen while capturing is unreachable in vLLM by construction;
    hitting it means an unvalidated scale reached the captured path, and
    the gate fails closed instead of syncing (which would abort the
    capture) or skipping (which would silently drop the check).
    """
    if scale is None:
        return
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError(
                f"q4 KV milestone 1 implements the unit global scale only; "
                f"{name} must be None or a 1-element unit tensor"
            )
        key = (str(scale.device), scale.data_ptr(), scale._version)
        entry = _VALIDATED_SCALES.get(key)
        if entry is not None and entry() is scale:
            return  # this exact object was validated eagerly before
        if entry is not None and entry() is None:
            del _VALIDATED_SCALES[key]  # dead entry; never an approval
        if _stream_is_capturing():
            raise RuntimeError(
                f"q4 KV unit-scale gate: {name} was never validated in an "
                "eager forward and is first seen while a CUDA graph is "
                "capturing; a host sync here aborts the capture. vLLM "
                "always warm-up-runs every captured step eagerly first "
                "(cudagraph_utils.py:464-467), so a cache miss under "
                "capture means an unvalidated scale reached the captured "
                "path -- refusing."
            )
        if float(scale.reshape(-1)[0].item()) != 1.0:
            raise ValueError(
                f"q4 KV milestone 1 implements the unit global scale only; "
                f"{name} must be None or 1.0"
            )
        _VALIDATED_SCALES[key] = weakref.ref(scale)
        return
    if not isinstance(scale, (bool, int, float)) or scale != 1.0:
        raise ValueError(
            f"q4 KV milestone 1 implements the unit global scale only; "
            f"{name} must be None or 1.0, got {scale!r}"
        )


def _py_scalar(name: str, value: object, *types: type) -> Any:
    """Fail closed unless a scalar argument is a plain Python number.

    ``bool``/``int``/``float`` on a torch.Tensor is a host sync, which
    CUDA forbids under graph capture: a tensor in one of these slots must
    be refused, never coerced. The vendor op's signature carries these as
    Python scalars, so the check only rejects what would sync.
    """
    if isinstance(value, types) and not isinstance(value, torch.Tensor):
        return value
    raise TypeError(
        f"q4 KV refused: {name} must be a plain Python "
        f"{' or '.join(t.__name__ for t in types)}, got "
        f"{type(value).__name__} (coercing it would host-sync, which is "
        "forbidden during CUDA-graph capture)"
    )


def _q4_pool_bytes(pool: torch.Tensor) -> torch.Tensor:
    """The bound q4 container as raw uint8 bytes (fail closed otherwise).

    vLLM's Triton impl views ANY quantized pool as its fp8 dtype before the
    reader hooks -- ``use_fp8_kv = is_quantized_kv_cache("nvfp4")`` is True
    (vllm/models/minimax_m3/common/sparse_attention.py:465 with
    vllm/v1/kv_cache_interface.py:116), and the impl's forward re-views the
    pool at sparse_attention.py:480. The bound container is one byte per
    element either way; re-view it to uint8 (same storage) so the
    byte-level kernels see raw bytes. A wider dtype cannot be the bound
    container and is refused.
    """
    if pool.dtype is torch.uint8:
        return pool
    if pool.element_size() != 1:
        raise ValueError(
            "q4 pool elements must be 1 byte; refusing to re-view "
            f"dtype {pool.dtype}"
        )
    return pool.view(torch.uint8)


# ---------------------------------------------------------------------------
# Backend: nvfp4 spec arithmetic off SM100
# ---------------------------------------------------------------------------


def _build_backend_cls():
    from vllm.models.minimax_m3.common.sparse_attention import (
        MiniMaxM3SparseBackend,
    )
    from vllm.utils.torch_utils import (
        get_dtype_size,
        nvfp4_kv_cache_full_dim,
    )
    from vllm.v1.kv_cache_interface import KVCacheLayout

    class MiniMaxM3SparseQ4Backend(MiniMaxM3SparseBackend):
        """Main-MSA backend over our q4 (nvfp4-layout) KV pages."""

        supported_kv_cache_dtypes: list = ["nvfp4"]

        @staticmethod
        def get_name() -> str:
            return "MINIMAX_M3_SPARSE_Q4"

        @classmethod
        def customize_spec(cls, spec):
            # Mirror the upstream nvfp4 arithmetic (see the SM100 backend):
            # every head's K and V are two slots of packed data + block
            # scales, so the pool is 2*H*72 B per 128-token page.
            if (
                spec.state_content_bytes is not None
                or not spec.kv_quant_mode.is_nvfp4
            ):
                return spec
            return replace(
                spec,
                num_head_slots=2 * spec.num_kv_heads,
                state_content_bytes=(
                    nvfp4_kv_cache_full_dim(spec.head_size)
                    * get_dtype_size(spec.dtype)
                ),
                kv_head_major_slots=True,
            )

        @classmethod
        def supported_kv_cache_layouts(cls):
            # Every head's K/V slots are one contiguous run of the page.
            return (KVCacheLayout.LBHNC,)

    return MiniMaxM3SparseQ4Backend


def _route_q4(kv_cache_dtype: str) -> bool:
    """nvfp4 on non-SM100/non-ROCm CUDA (GB10/sm_121) is ours to serve."""
    from vllm.v1.kv_cache_interface import get_kv_quant_mode
    from vllm.platforms import current_platform

    if not get_kv_quant_mode(kv_cache_dtype).is_nvfp4:
        return False
    if current_platform.is_rocm():
        return False  # the AMD nvfp4 backend serves ROCm
    if not current_platform.is_cuda():
        raise ValueError(
            "The MiniMax-M3 q4 KV overlay requires CUDA (GB10/sm_121); "
            f"kv_cache_dtype={kv_cache_dtype!r} is not supported here"
        )
    if current_platform.is_device_capability_family(100):
        return False  # SM100 serves nvfp4 with the vendor MSA backend
    return True


def _q4_backend_cls():
    """Build the overlay backend class lazily (once), at first q4 routing.

    Building at install time would drag ``vllm.utils.torch_utils`` /
    ``vllm.v1.kv_cache_interface`` in eagerly; the class is only needed on
    the q4 route, and the lazy import fails loudly there if unavailable.
    """
    global _Q4_BACKEND_CLS
    if _Q4_BACKEND_CLS is None:
        _Q4_BACKEND_CLS = _build_backend_cls()
    return _Q4_BACKEND_CLS


_Q4_BACKEND_CLS = None


def _make_select_main(original):
    def select_main_backend_and_impl_cls(
        *,
        topk_blocks: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        emits_sparse_block_table: bool = False,
    ):
        if _route_q4(kv_cache_dtype):
            from vllm.models.minimax_m3.common.sparse_attention import (
                MiniMaxM3SparseTritonImpl,
            )

            logger.info_once(
                "MiniMax M3 sparse attention selected the q4 overlay "
                "(kv_cache_dtype=%s, topk_blocks=%s)",
                kv_cache_dtype,
                topk_blocks,
            )
            return _q4_backend_cls(), MiniMaxM3SparseTritonImpl
        return original(
            topk_blocks=topk_blocks,
            kv_cache_dtype=kv_cache_dtype,
            num_kv_heads=num_kv_heads,
            emits_sparse_block_table=emits_sparse_block_table,
        )

    return select_main_backend_and_impl_cls


# ---------------------------------------------------------------------------
# W1: cache-update op wrapper
# ---------------------------------------------------------------------------

# Positional order of the vendor op's Python signature (vllm/_custom_ops.py:
# fused_minimax_m3_qknorm_rope_kv_insert); keyword overrides follow.
_W1_PARAMS = (
    "qkv",
    "q_norm_weight",
    "k_norm_weight",
    "cos_sin_cache",
    "positions",
    "num_heads",
    "num_kv_heads",
    "rotary_dim",
    "eps",
    "index_q_norm_weight",
    "index_k_norm_weight",
    "num_index_heads",
    "slot_mapping",
    "index_slot_mapping",
    "kv_cache",
    "index_cache",
    "block_size",
    "q_out",
    "index_q_out",
    "kv_cache_dtype",
    "skip_index_branch",
    "q_fp8_out",
    "q_fp8_scale",
    "kv_k_scale",
    "kv_v_scale",
    "write_qkv",
)


def _make_w1_wrapper(original):
    def fused_minimax_m3_qknorm_rope_kv_insert(*args, **kwargs):
        bound = dict(zip(_W1_PARAMS, args))
        for key, value in kwargs.items():
            if key not in _W1_PARAMS:
                raise TypeError(f"unknown W1 argument {key!r}")
            bound[key] = value
        pool = bound.get("kv_cache")
        if not _is_q4_pool(pool):
            return original(*args, **kwargs)
        pool = _q4_pool_bytes(pool)
        # ---- q4 route (fail closed) ----
        if bound.get("kv_cache_dtype") != "nvfp4":
            raise ValueError(
                "q4 pool with kv_cache_dtype="
                f"{bound.get('kv_cache_dtype')!r}: refusing"
            )
        if bound.get("q_fp8_out") is not None:
            raise ValueError("the q4 overlay does not serve fp8 queries")
        if not _py_scalar(
            "write_qkv", bound.get("write_qkv", True), bool
        ):
            raise ValueError("the q4 overlay requires write_qkv=True")
        q_out = bound.get("q_out")
        if q_out is None:
            raise ValueError("the q4 overlay requires a bf16 q_out buffer")
        _assert_unit_scale("kv_k_scale", bound.get("kv_k_scale"))
        _assert_unit_scale("kv_v_scale", bound.get("kv_v_scale"))
        num_heads = _py_scalar("num_heads", bound["num_heads"], int)
        num_kv_heads = _py_scalar(
            "num_kv_heads", bound["num_kv_heads"], int
        )
        rotary_dim = _py_scalar("rotary_dim", bound["rotary_dim"], int)
        eps = _py_scalar("eps", bound["eps"], int, float)
        if pool.shape[1] != 2 * num_kv_heads:
            raise ValueError("q4 pool slot count mismatch")
        skip_index = _py_scalar(
            "skip_index_branch", bound.get("skip_index_branch", False), bool
        )
        if skip_index and (
            bound.get("index_q_out") is not None
            or bound.get("index_cache") is not None
        ):
            raise ValueError(
                "skip_index_branch with index outputs is not supported"
            )
        num_index_heads = (
            0 if skip_index else _py_scalar(
                "num_index_heads", bound["num_index_heads"], int
            )
        )
        index_q_out = None if skip_index else bound.get("index_q_out")
        index_cache = None if skip_index else bound.get("index_cache")
        from vllm_exl3 import minimax_q4kv_kernels as kernels

        kernels.q4_w1_insert(
            bound["qkv"],
            bound["q_norm_weight"],
            bound["k_norm_weight"],
            bound["cos_sin_cache"],
            bound["positions"],
            num_heads,
            num_kv_heads,
            rotary_dim,
            float(eps),
            None if skip_index else bound.get("index_q_norm_weight"),
            None if skip_index else bound.get("index_k_norm_weight"),
            num_index_heads,
            bound.get("slot_mapping"),
            bound.get("index_slot_mapping"),
            pool,
            index_cache,
            _py_scalar("block_size", bound.get("block_size", 0), int),
            q_out,
            index_q_out,
        )

    return fused_minimax_m3_qknorm_rope_kv_insert


# ---------------------------------------------------------------------------
# R2: decode reader wrapper; prefill staging wrapper
# ---------------------------------------------------------------------------


def _make_decode_wrapper(original):
    def minimax_m3_sparse_attn_decode(
        q,
        kv_cache,
        topk_idx,
        block_table,
        seq_lens,
        num_kv_heads,
        sm_scale,
        output,
        decode_query_len,
        k_scale=None,
        v_scale=None,
    ):
        if not _is_q4_pool(kv_cache):
            return original(
                q,
                kv_cache,
                topk_idx,
                block_table,
                seq_lens,
                num_kv_heads,
                sm_scale,
                output,
                decode_query_len,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        kv_cache = _q4_pool_bytes(kv_cache)
        _assert_unit_scale("k_scale", k_scale)
        _assert_unit_scale("v_scale", v_scale)
        num_kv_heads = _py_scalar("num_kv_heads", num_kv_heads, int)
        sm_scale = _py_scalar("sm_scale", sm_scale, int, float)
        decode_query_len = _py_scalar(
            "decode_query_len", decode_query_len, int
        )
        from vllm_exl3 import minimax_q4kv_kernels as kernels

        kernels.q4_sparse_decode(
            q,
            kv_cache,
            topk_idx,
            block_table,
            seq_lens,
            num_kv_heads,
            sm_scale,
            output,
            decode_query_len,
        )

    return minimax_m3_sparse_attn_decode


def _prefill_fast_enabled() -> bool:
    """VLLM_EXL3_Q4KV_PREFILL_FAST: single-pass q4 prefill staging.

    Default off (0): the milestone-1 staging path, byte for byte. On (1):
    :func:`vllm_exl3.minimax_q4kv_reference.plan_prefill_slices_fast` +
    ``q4_stage_pages``, which produce the SAME slice plan and the SAME
    staged bf16 bytes at ~8x less memory traffic and O(1) host syncs per
    slice instead of one per row. Read per call (an ``os.environ`` lookup
    on the prefill path, which never runs under capture).

    Any other value fails closed: a typo must not silently select a path
    the operator did not ask for.
    """
    raw = os.environ.get("VLLM_EXL3_Q4KV_PREFILL_FAST")
    if raw is None:
        return False
    val = raw.strip().lower()
    if val in ("1", "true", "on", "yes"):
        return True
    if val in ("0", "false", "off", "no", ""):
        return False
    raise ValueError(
        "VLLM_EXL3_Q4KV_PREFILL_FAST must be 0 or 1, got " f"{raw!r}"
    )


def _make_prefill_wrapper(original):
    def minimax_m3_sparse_attn(
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        num_kv_heads,
        sm_scale,
        output,
        k_scale=None,
        v_scale=None,
    ):
        if not _is_q4_pool(kv_cache):
            return original(
                q,
                kv_cache,
                topk_idx,
                block_table,
                cu_seqlens_q,
                seq_lens,
                prefix_lens,
                max_query_len,
                num_kv_heads,
                sm_scale,
                output,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        kv_cache = _q4_pool_bytes(kv_cache)
        _assert_unit_scale("k_scale", k_scale)
        _assert_unit_scale("v_scale", v_scale)
        # The prefill wrapper is never captured under FULL_DECODE_ONLY
        # (see the module docstring), but the scalar slots get the same
        # fail-closed gate so a tensor can never reach a coercion.
        num_kv_heads = _py_scalar("num_kv_heads", num_kv_heads, int)
        sm_scale = _py_scalar("sm_scale", sm_scale, int, float)
        max_query_len = _py_scalar(
            "max_query_len", max_query_len, int
        )
        # Fail-closed device gate, exactly as the W1/R2 kernel wrappers:
        # the boot18 crash class is a CPU tensor reaching the serve path
        # (there it was the dequant scale-offset table; the gate must
        # name every offender before any staging math runs).
        from vllm_exl3 import minimax_q4kv_kernels as q4kernels

        q4kernels._device_gate(
            kv_cache,
            "kv_cache",
            (
                ("q", q),
                ("topk_idx", topk_idx),
                ("block_table", block_table),
                ("cu_seqlens_q", cu_seqlens_q),
                ("seq_lens", seq_lens),
                ("prefix_lens", prefix_lens),
                ("output", output),
            ),
        )
        # Memory-bounded staging: one grow-once bf16 buffer per (device,
        # head count), filled per slice under the page cap; the chunk is
        # sliced so every call's staged union fits the cap and the result
        # equals the uncapped staging (see the reference module).
        #
        # VLLM_EXL3_Q4KV_PREFILL_FAST=1 swaps both halves of the staging
        # path for their single-pass equivalents (same slice plan, same
        # staged bytes, see the two docstrings below). Default off: the
        # GPU timing gate in tools/gb10_q4kv_parity.py must run first.
        fast = _prefill_fast_enabled()
        if fast:
            plan = ref.plan_prefill_slices_fast
            fill = q4kernels.q4_stage_pages
        else:
            plan = ref.plan_prefill_slices
            fill = ref.fill_staging_buffer
        max_pages = ref.staging_max_pages(num_kv_heads)
        slices = plan(
            kv_cache, block_table, topk_idx, cu_seqlens_q,
            num_kv_heads, max_pages,
        )
        needed = max((s["n"] for s in slices), default=0)
        buffer = _staging_buffer(
            kv_cache.device, max(max_pages, needed), num_kv_heads
        )
        num_reqs = cu_seqlens_q.shape[0] - 1
        full_batch = (
            len(slices) == 1
            and slices[0]["reqs"] == list(range(num_reqs))
            and slices[0]["q"] == (0, q.shape[0])
        )
        if full_batch:
            # The whole chunk fits the cap: one call, the milestone-1
            # behavior with every non-pool argument passed through
            # unchanged (the staged pool view replaces the pool, the
            # fake table replaces the block table).
            n = fill(buffer, kv_cache, slices[0]["pages"])
            fake_bt = ref.slice_fake_block_table(
                block_table, slices[0]["reqs"], slices[0]["pages"], n,
                kv_cache.shape[0],
            )
            return original(
                q,
                buffer[:n + 1],
                topk_idx,
                fake_bt,
                cu_seqlens_q,
                seq_lens,
                prefix_lens,
                max_query_len,
                num_kv_heads,
                sm_scale,
                output,
            )
        cu_list = cu_seqlens_q.reshape(-1).to(torch.int64).tolist()
        prefix_list = prefix_lens.reshape(-1).to(torch.int64).tolist()
        for s in slices:
            qs, qe = s["q"]
            n = fill(buffer, kv_cache, s["pages"])
            fake_bt = ref.slice_fake_block_table(
                block_table, s["reqs"], s["pages"], n, kv_cache.shape[0]
            )
            cu_new, prefix_new, max_q = ref.prefill_slice_metadata(
                s, cu_list, prefix_list, kv_cache.device
            )
            original(
                q[qs:qe],
                buffer[:n + 1],
                topk_idx[:, qs:qe, :],
                fake_bt,
                cu_new,
                seq_lens[s["reqs"][0]:s["reqs"][-1] + 1],
                prefix_new,
                max_q,
                num_kv_heads,
                sm_scale,
                output[qs:qe],
            )
        return None

    return minimax_m3_sparse_attn


# One preallocated bf16 staging buffer per (device, head count): the
# grow-once rule -- allocated at the configured capacity on first use,
# grown only when a hard-floor slice (a single query whose own selection
# exceeds the cap) needs more, never shrunk and never reallocated per
# call. The per-call peak is therefore the configured cap plus the
# bounded dequant transient, independent of the chunk's selected-page
# union.
_STAGING_BUFFERS: dict = {}


def _staging_buffer(
    device: torch.device, capacity_pages: int, num_kv_heads: int
) -> torch.Tensor:
    key = (device, num_kv_heads)
    buf = _STAGING_BUFFERS.get(key)
    if buf is None or buf.shape[0] < capacity_pages + 1:
        new_cap = capacity_pages + 1
        if buf is not None:
            new_cap = max(new_cap, buf.shape[0])
        buf = torch.zeros(
            (new_cap, num_kv_heads, _SPARSE_BLOCK_SIZE, 2 * ref.HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        )
        _STAGING_BUFFERS[key] = buf
    return buf


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

# Attribute that marks every overlay wrapper (install idempotency + the
# fail-closed self-check).
_Q4KV_WRAPPER_MARKER = "__vllm_exl3_q4kv_wrapper__"

# (module path, attribute name, hook) for every binding of the four hooked
# callables. REQUIRED sites exist on every platform the overlay can serve;
# an unpatched one is exactly the boot17 failure mode (the engine keeps the
# upstream SM100-only selector), so import failure or a missing attribute
# is a hard error. Verified against the current vllm-image:
#   - common/ops/__init__.py:11 re-exports the reader hooks by value;
#   - common/sparse_attention.py:35/:40 imports them by value (common.ops
#     off ROCm, amd.ops on ROCm) and :525 defines the selector;
#   - nvidia/model.py:128 imports the selector BY VALUE (boot17);
#   - amd/model.py:133 imports the selector BY VALUE;
#   - nvidia/sparse_attention_msa.py:29 imports the decode hook by value;
#   - amd/ops/sparse_attn.py:18 re-exports the decode hook by value.
# The W1 op is called only as ``ops.<name>`` on the ``vllm._custom_ops``
# module object (nvidia/model.py:22, amd/model.py:26, dspark.py:16), so the
# single module-attribute patch covers every call site.
_REQUIRED_BINDING_SITES = (
    ("vllm._custom_ops", "fused_minimax_m3_qknorm_rope_kv_insert", "w1"),
    (
        "vllm.models.minimax_m3.common.ops.sparse_attn",
        "minimax_m3_sparse_attn",
        "prefill",
    ),
    (
        "vllm.models.minimax_m3.common.ops.sparse_attn",
        "minimax_m3_sparse_attn_decode",
        "decode",
    ),
    (
        "vllm.models.minimax_m3.common.ops",
        "minimax_m3_sparse_attn",
        "prefill",
    ),
    (
        "vllm.models.minimax_m3.common.ops",
        "minimax_m3_sparse_attn_decode",
        "decode",
    ),
    (
        "vllm.models.minimax_m3.common.sparse_attention",
        "select_main_backend_and_impl_cls",
        "selector",
    ),
    (
        "vllm.models.minimax_m3.common.sparse_attention",
        "minimax_m3_sparse_attn",
        "prefill",
    ),
    (
        "vllm.models.minimax_m3.common.sparse_attention",
        "minimax_m3_sparse_attn_decode",
        "decode",
    ),
    (
        "vllm.models.minimax_m3.nvidia.model",
        "select_main_backend_and_impl_cls",
        "selector",
    ),
)

# Platform-specific by-value bindings: the SM100-only MSA module and the
# ROCm-only AMD modules. The q4 overlay never routes on SM100 or ROCm
# (_route_q4 returns False there), so an unpatched binding is
# behavior-neutral; patch them whenever the module is importable here and
# log the ones that are not.
_OPTIONAL_BINDING_SITES = (
    (
        "vllm.models.minimax_m3.nvidia.sparse_attention_msa",
        "minimax_m3_sparse_attn_decode",
        "decode",
    ),
    (
        "vllm.models.minimax_m3.amd.model",
        "select_main_backend_and_impl_cls",
        "selector",
    ),
    (
        "vllm.models.minimax_m3.amd.ops.sparse_attn",
        "minimax_m3_sparse_attn",
        "prefill",
    ),
    (
        "vllm.models.minimax_m3.amd.ops.sparse_attn",
        "minimax_m3_sparse_attn_decode",
        "decode",
    ),
)

# The defining module of each hook: where the ORIGINAL callable lives.
_HOOK_DEFINING_SITES = {
    "selector": (
        "vllm.models.minimax_m3.common.sparse_attention",
        "select_main_backend_and_impl_cls",
    ),
    "w1": ("vllm._custom_ops", "fused_minimax_m3_qknorm_rope_kv_insert"),
    "decode": (
        "vllm.models.minimax_m3.common.ops.sparse_attn",
        "minimax_m3_sparse_attn_decode",
    ),
    "prefill": (
        "vllm.models.minimax_m3.common.ops.sparse_attn",
        "minimax_m3_sparse_attn",
    ),
}

_HOOK_FACTORIES = {
    "selector": _make_select_main,
    "w1": _make_w1_wrapper,
    "decode": _make_decode_wrapper,
    "prefill": _make_prefill_wrapper,
}


def _import_binding_module(path: str):
    """Import a binding module (seam; the CPU tests substitute fakes)."""
    import importlib

    return importlib.import_module(path)


def _is_q4kv_wrapper(fn: object) -> bool:
    return getattr(fn, _Q4KV_WRAPPER_MARKER, False) is True


def _make_hook_wrapper(hook: str, original):
    fn = _HOOK_FACTORIES[hook](original)
    setattr(fn, _Q4KV_WRAPPER_MARKER, True)
    setattr(fn, "__q4kv_original__", original)
    return fn


def install() -> None:
    """Patch every binding of the four hook points, robust to import order.

    Upstream modules import the hooks BY VALUE, so patching only the
    defining module's attribute never reaches an importer that was already
    loaded (the boot17 failure: ``models/minimax_m3/nvidia/model.py:128``
    kept the original SM100-only selector). ``install`` therefore

    1. imports every binding module itself (required sites must exist);
    2. patches the attribute on EVERY binding module -- the defining
       modules, the re-exporting ``common.ops`` package, and the by-value
       importers (``nvidia/model.py``, ``amd/model.py``, the SM100 MSA and
       ROCm AITER modules whenever those are importable here);
    3. logs one INFO line listing every patched binding;
    4. fails closed: if any importable binding site is not the overlay
       wrapper at the end, raises naming the unpatched site.

    Platform-specific modules that cannot be imported here (the SM100-only
    ``nvidia.sparse_attention_msa`` on GB10, the ROCm-only ``amd.*`` modules
    on CUDA) are logged as unpatched and skipped: the q4 overlay never
    routes on SM100 or ROCm, so their bindings stay upstream there.
    Idempotent; raises if vllm is missing.
    """
    # Import each binding module once, required sites first.
    loaded: dict = {}
    unavailable: list = []
    for path, name, hook in _REQUIRED_BINDING_SITES + _OPTIONAL_BINDING_SITES:
        if path in loaded or path in unavailable:
            continue
        try:
            loaded[path] = _import_binding_module(path)
        except ImportError as exc:
            if (path, name, hook) in _REQUIRED_BINDING_SITES:
                raise RuntimeError(
                    "q4 KV overlay: required binding module "
                    f"{path!r} is not importable ({exc}); refusing to "
                    "install a partial overlay (the engine would keep the "
                    "upstream SM100-only nvfp4 selector)"
                ) from exc
            unavailable.append(path)

    # One wrapper per hook, built from the ORIGINAL callable of the
    # defining module (never from a previously-installed wrapper).
    wrappers: dict = {}
    for hook, (path, name) in _HOOK_DEFINING_SITES.items():
        original = getattr(loaded[path], name, None)
        if original is None:
            raise RuntimeError(
                f"q4 KV overlay: defining module {path!r} no longer "
                f"exposes {name!r}; upstream vLLM has drifted -- refusing"
            )
        if _is_q4kv_wrapper(original):
            wrappers[hook] = original  # already installed; keep as-is
        else:
            wrappers[hook] = _make_hook_wrapper(hook, original)

    # Patch every binding site.
    patched: list = []
    for path, name, hook in _REQUIRED_BINDING_SITES + _OPTIONAL_BINDING_SITES:
        if path in unavailable:
            continue
        current = getattr(loaded[path], name, None)
        if current is None:
            raise RuntimeError(
                f"q4 KV overlay: binding site {path}.{name} is missing; "
                "upstream vLLM has drifted -- refusing"
            )
        setattr(loaded[path], name, wrappers[hook])
        patched.append(f"{path}.{name}")

    logger.info(
        "vllm-exl3 q4 KV overlay installed (VLLM_EXL3_Q4KV=1); patched "
        "bindings: %s%s",
        ", ".join(patched),
        (
            "; platform-specific modules not importable here (left to "
            "upstream, the q4 overlay never routes there): "
            + ", ".join(sorted({f"{p}.{n}" for p, n, _ in
                                _OPTIONAL_BINDING_SITES
                                if p in unavailable}))
        )
        if unavailable
        else "",
    )

    # Fail-closed self-check: every importable binding must be ours.
    for path, name, _hook in _REQUIRED_BINDING_SITES + _OPTIONAL_BINDING_SITES:
        if path in unavailable:
            continue
        fn = getattr(loaded[path], name, None)
        if not _is_q4kv_wrapper(fn):
            raise RuntimeError(
                f"q4 KV overlay self-check failed: {path}.{name} is not "
                "the overlay wrapper -- the q4 KV path would silently fall "
                "back to the upstream implementation. Refusing to continue."
            )
