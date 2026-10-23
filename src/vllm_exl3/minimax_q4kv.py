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

All wrappers fail closed: they raise on any unsupported combination rather
than silently miscomputing.
"""

from __future__ import annotations

import logging
from dataclasses import replace

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


def _assert_unit_scale(name: str, scale: object) -> None:
    if scale is None:
        return
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1 or float(scale.reshape(-1)[0].item()) != 1.0:
            raise ValueError(
                f"q4 KV milestone 1 implements the unit global scale only; "
                f"{name} must be None or 1.0"
            )
        return
    if scale != 1.0:
        raise ValueError(
            f"q4 KV milestone 1 implements the unit global scale only; "
            f"{name} must be None or 1.0, got {scale}"
        )


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


def _make_select_main(original):
    backend_cls = _build_backend_cls()

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
            return backend_cls, MiniMaxM3SparseTritonImpl
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
        # ---- q4 route (fail closed) ----
        if bound.get("kv_cache_dtype") != "nvfp4":
            raise ValueError(
                "q4 pool with kv_cache_dtype="
                f"{bound.get('kv_cache_dtype')!r}: refusing"
            )
        if bound.get("q_fp8_out") is not None:
            raise ValueError("the q4 overlay does not serve fp8 queries")
        if not bound.get("write_qkv", True):
            raise ValueError("the q4 overlay requires write_qkv=True")
        q_out = bound.get("q_out")
        if q_out is None:
            raise ValueError("the q4 overlay requires a bf16 q_out buffer")
        _assert_unit_scale("kv_k_scale", bound.get("kv_k_scale"))
        _assert_unit_scale("kv_v_scale", bound.get("kv_v_scale"))
        if pool.shape[1] != 2 * int(bound["num_kv_heads"]):
            raise ValueError("q4 pool slot count mismatch")
        skip_index = bool(bound.get("skip_index_branch", False))
        if skip_index and (
            bound.get("index_q_out") is not None
            or bound.get("index_cache") is not None
        ):
            raise ValueError(
                "skip_index_branch with index outputs is not supported"
            )
        num_index_heads = 0 if skip_index else int(bound["num_index_heads"])
        index_q_out = None if skip_index else bound.get("index_q_out")
        index_cache = None if skip_index else bound.get("index_cache")
        from vllm_exl3 import minimax_q4kv_kernels as kernels

        kernels.q4_w1_insert(
            bound["qkv"],
            bound["q_norm_weight"],
            bound["k_norm_weight"],
            bound["cos_sin_cache"],
            bound["positions"],
            int(bound["num_heads"]),
            int(bound["num_kv_heads"]),
            int(bound["rotary_dim"]),
            float(bound["eps"]),
            None if skip_index else bound.get("index_q_norm_weight"),
            None if skip_index else bound.get("index_k_norm_weight"),
            num_index_heads,
            bound.get("slot_mapping"),
            bound.get("index_slot_mapping"),
            pool,
            index_cache,
            int(bound.get("block_size", 0)),
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
        _assert_unit_scale("k_scale", k_scale)
        _assert_unit_scale("v_scale", v_scale)
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
        _assert_unit_scale("k_scale", k_scale)
        _assert_unit_scale("v_scale", v_scale)
        # Milestone-1 staging: dequantize the selected physical pages to
        # bf16 and run the original prefill kernel on the staged pool.
        staging, fake_block_table = ref.build_prefill_staging(
            kv_cache, block_table, topk_idx, num_kv_heads
        )
        return original(
            q,
            staging,
            topk_idx,
            fake_block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            max_query_len,
            num_kv_heads,
            sm_scale,
            output,
        )

    return minimax_m3_sparse_attn


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def install() -> None:
    """Patch the four hook points. Idempotent; raises if vllm is missing."""
    import vllm._custom_ops as ops
    from vllm.models.minimax_m3.common import sparse_attention
    from vllm.models.minimax_m3.common.ops import sparse_attn

    # 1. Backend/impl selection: un-raise nvfp4 off SM100 and route to the
    #    overlay backend (spec arithmetic + [128] block sizes unchanged).
    sparse_attention.select_main_backend_and_impl_cls = (
        _make_select_main(sparse_attention.select_main_backend_and_impl_cls)
    )
    # select_main_impl_cls delegates to the module-global selector above, so
    # it picks up the patch; keep its attribute intact for direct callers.

    # 2. Cache update: q4 pool -> W1; everything else -> original op.
    ops.fused_minimax_m3_qknorm_rope_kv_insert = _make_w1_wrapper(
        ops.fused_minimax_m3_qknorm_rope_kv_insert
    )

    # 3/4. Readers: q4 pool -> R2 / staged prefill; else original kernels.
    # Patch both the ops module and the impl module's imported names.
    decode_wrapper = _make_decode_wrapper(
        sparse_attn.minimax_m3_sparse_attn_decode
    )
    prefill_wrapper = _make_prefill_wrapper(
        sparse_attn.minimax_m3_sparse_attn
    )
    sparse_attn.minimax_m3_sparse_attn_decode = decode_wrapper
    sparse_attn.minimax_m3_sparse_attn = prefill_wrapper
    sparse_attention.minimax_m3_sparse_attn_decode = decode_wrapper
    sparse_attention.minimax_m3_sparse_attn = prefill_wrapper
