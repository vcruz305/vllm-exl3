"""VLLM_EXL3_INDEXER_MXFP8: native-MXFP8 retention of the MSA indexer shards.

CPU-only (no vLLM, no GPU, no kernel launch):

- (a) default-off is byte-identical to the pre-switch loader: the stashed
  MXFP8 payloads are dequantized into the same BF16 staging rows and no
  MXFP8 state appears on the layer;
- (b) the retained (F8_E4M3 payload, E8M0 block-32 scales) pair is
  bit-preserving, and the emulated MXFP8 GEMM semantics (what the kernel
  consumes, through the F8_128x4 scale swizzle its load step applies)
  match a pure-torch dequant+matmul reference on random data;
- (c) memory accounting: the switch-on parameters are fp8+uint8 and their
  total bytes are ~half of the BF16 staging at the real pack shapes
  (60 layers: 450 MiB BF16 -> ~232 MiB retained);
- (d) the switch-on path fails closed when vLLM's MXFP8 kernel is
  unavailable, and apply() routes the two indexer slices through the
  kernel with output shape/dtype identical to the BF16 path.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

import vllm_exl3.exl3 as exl3

# Small stand-in geometry for the fused [q | k | v | index_q | index_k]
# layer: input 64 (two 32-wide scale blocks), indexer outs (32, 16).
IN = 64
OUTS = [64, 64, 64, 32, 16]
INDEXER_SHARDS = (3, 4)
INDEXER_OUTS = OUTS[3:]

FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "minimax_m3_layer0_real_shards.json",
)

# The census layer count of the sparse-attention (indexer-carrying) layers.
INDEXER_LAYERS = 60


def _finite_e4m3_bytes(shape) -> torch.Tensor:
    """Random F8_E4M3 payload bytes without the 0x7F/0xFF NaN encodings."""
    b = torch.randint(0, 239, shape, dtype=torch.int32)
    b = b + (b >= 0x7F).to(torch.int32)  # {0..126, 128..238}
    return b.to(torch.uint8)


def _scales(shape) -> torch.Tensor:
    return torch.randint(
        100, 130, shape, dtype=torch.int32
    ).to(torch.uint8)


def _method() -> exl3.Exl3LinearMethod:
    return exl3.Exl3LinearMethod.__new__(exl3.Exl3LinearMethod)


def _indexer_stash():
    """Two synthetic indexer shards exactly as the loaders stash them."""
    payloads = [
        _finite_e4m3_bytes((n, IN)) for n in INDEXER_OUTS
    ]
    scales = [_scales((n, IN // 32)) for n in INDEXER_OUTS]
    return payloads, scales


def _fake_layer(payloads, scales, *, bf16_staging: bool):
    layer = SimpleNamespace()
    layer._exl3_indexer_shards = INDEXER_SHARDS
    layer._exl3_indexer_payloads = list(payloads)
    layer._exl3_indexer_scales = list(scales)
    if bf16_staging:
        # The pre-switch loader allocates the BF16 staging rows the dequant
        # writes into (weight param of the fused layer).
        layer.weight = torch.nn.Parameter(
            torch.zeros(sum(INDEXER_OUTS), IN, dtype=torch.bfloat16),
            requires_grad=False,
        )
    return layer


# ---------------------------------------------------------------------------
# (a) switch parsing + default-off byte-identity
# ---------------------------------------------------------------------------


def test_switch_is_off_by_default_and_env_parsed(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_INDEXER_MXFP8", raising=False)
    assert exl3._indexer_mxfp8_enabled() is False
    monkeypatch.setenv("VLLM_EXL3_INDEXER_MXFP8", "0")
    assert exl3._indexer_mxfp8_enabled() is False
    monkeypatch.setenv("VLLM_EXL3_INDEXER_MXFP8", "1")
    assert exl3._indexer_mxfp8_enabled() is True
    monkeypatch.setenv("VLLM_EXL3_INDEXER_MXFP8", "true")
    assert exl3._indexer_mxfp8_enabled() is True


def test_default_off_dequant_staging_is_byte_identical(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_INDEXER_MXFP8", raising=False)
    payloads, scales = _indexer_stash()
    layer = _fake_layer(payloads, scales, bf16_staging=True)
    # The pre-switch reference: dequant_minimax_indexer into the staging
    # rows, in shard order, at the bf16 shard offsets.
    reference = torch.cat(
        [exl3.dequant_minimax_indexer(p, s) for p, s in zip(payloads, scales)]
    )
    method = _method()
    method._process_indexer_shards(
        layer, OUTS, list(INDEXER_SHARDS), INDEXER_SHARDS, False
    )
    assert torch.equal(layer.weight.data, reference)
    # The stash is consumed and no MXFP8 state appears on the layer.
    assert layer._exl3_indexer_payloads == [None, None]
    assert layer._exl3_indexer_scales == [None, None]
    assert not hasattr(layer, "_exl3_indexer_mxfp8")


def test_env_zero_behaves_like_default_off(monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_INDEXER_MXFP8", "0")
    payloads, scales = _indexer_stash()
    layer = _fake_layer(payloads, scales, bf16_staging=True)
    method = _method()
    method._process_indexer_shards(
        layer, OUTS, list(INDEXER_SHARDS), INDEXER_SHARDS,
        bool(INDEXER_SHARDS) and exl3._indexer_mxfp8_enabled(),
    )
    assert torch.equal(
        layer.weight.data,
        torch.cat(
            [exl3.dequant_minimax_indexer(p, s)
             for p, s in zip(payloads, scales)]
        ),
    )
    assert not hasattr(layer, "_exl3_indexer_mxfp8")


def test_default_off_incomplete_stash_still_raises(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_INDEXER_MXFP8", raising=False)
    payloads, scales = _indexer_stash()
    payloads[1] = None  # index_k payload never arrived
    layer = _fake_layer(payloads, scales, bf16_staging=True)
    with pytest.raises(RuntimeError, match="indexer shard 4 incomplete"):
        _method()._process_indexer_shards(
            layer, OUTS, list(INDEXER_SHARDS), INDEXER_SHARDS, False
        )


# ---------------------------------------------------------------------------
# (d) fail-closed + kernel binding
# ---------------------------------------------------------------------------


def test_switch_on_fails_closed_without_vllm_kernel(monkeypatch):
    # This VPS has no vllm and no GPU: the switch-on path must refuse to
    # run with a clear error naming the switch, never fall back to BF16.
    monkeypatch.setenv("VLLM_EXL3_INDEXER_MXFP8", "1")
    payloads, scales = _indexer_stash()
    layer = _fake_layer(payloads, scales, bf16_staging=False)
    with pytest.raises(RuntimeError, match="VLLM_EXL3_INDEXER_MXFP8"):
        _method()._process_indexer_shards(
            layer, OUTS, [], INDEXER_SHARDS, True
        )
    assert not hasattr(layer, "_exl3_indexer_mxfp8")


class _FakeKernel:
    """Stands in for vLLM's MXFP8 linear kernel on CPU.

    ``process_weights_after_loading`` records the call (the real one
    swizzles the scale block); ``apply_weights`` emulates the kernel's
    semantics -- dequantize the shim's own (fp8 payload, E8M0 scales)
    state and matmul -- so the glue can be compared against the BF16
    path without any GPU.
    """

    def __init__(self):
        self.processed = []
        self.applied = []

    def process_weights_after_loading(self, shim):
        self.processed.append(shim)

    def apply_weights(self, shim, x):
        self.applied.append((shim, x))
        w = exl3.dequant_minimax_indexer(
            shim.weight.data.view(torch.uint8), shim.weight_scale.data
        )
        return torch.nn.functional.linear(x, w)


def test_switch_on_binds_kernel_shims_and_frees_stash(monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_INDEXER_MXFP8", "1")
    payloads, scales = _indexer_stash()
    layer = _fake_layer(payloads, scales, bf16_staging=False)
    made = []

    def make(shape):
        k = _FakeKernel()
        made.append((shape, k))
        return k

    monkeypatch.setattr(exl3, "indexer_mxfp8_kernel", make)
    _method()._process_indexer_shards(
        layer, OUTS, [], INDEXER_SHARDS, True
    )
    shims = layer._exl3_indexer_mxfp8
    assert len(shims) == 2 and len(made) == 2
    # One shim per indexer shard, in shard order, with the checkpoint
    # geometry; the payload is the bitcast fp8 view and the scales stay
    # the row-major uint8 block the kernel's load step expects.
    for slot, (n, (kernel, shim)) in enumerate(zip(INDEXER_OUTS, shims)):
        assert made[slot][0] == (n, IN)
        assert kernel.processed == [shim]
        assert shim.weight.data.dtype == torch.float8_e4m3fn
        assert tuple(shim.weight.data.shape) == (n, IN)
        assert torch.equal(shim.weight.data.view(torch.uint8), payloads[slot])
        assert shim.weight_scale.data.dtype == torch.uint8
        assert tuple(shim.weight_scale.data.shape) == (n, IN // 32)
        assert torch.equal(shim.weight_scale.data, scales[slot])
        assert shim.params_dtype == torch.bfloat16
    assert layer._exl3_indexer_payloads == [None, None]
    assert layer._exl3_indexer_scales == [None, None]


def test_retention_rejects_bad_geometry():
    payload = _finite_e4m3_bytes((4, IN))
    with pytest.raises(ValueError, match="scales shape"):
        exl3.indexer_mxfp8_retain(payload, _scales((4, IN // 32 + 1)))
    with pytest.raises(ValueError, match="scale block"):
        exl3.indexer_mxfp8_retain(
            _finite_e4m3_bytes((4, IN + 1)), _scales((4, 2))
        )
    with pytest.raises(ValueError, match="uint8"):
        exl3.indexer_mxfp8_retain(
            payload, _scales((4, IN // 32)).to(torch.int32) & 0xFF
        )


# ---------------------------------------------------------------------------
# (b) emulated MXFP8 GEMM semantics vs a pure-torch reference
# ---------------------------------------------------------------------------


def _swizzle_reference(sf: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Test-local F8_128x4 scale swizzle, from the documented layout.

    The scale block [n, k // 32] is padded to 128-row tiles x 4-column
    tiles, viewed (n/128 tiles, 4, 32, k/128 tiles, 4) and transposed on
    the (4, k/128) axes -- the permutation vLLM's MXFP8 GEMM reads. The
    real kernel applies its own swizzle at load; this reference exists so
    the round trip through the swizzled order can be proven lossless on
    CPU. The real-kernel agreement is gated on the GB10 (parity case G).
    """
    scale_cols = k // 32
    mt = (n + 127) // 128
    ct = (scale_cols + 3) // 4
    padded = torch.zeros((mt * 128, ct * 4), dtype=sf.dtype)
    padded[:n, :scale_cols] = sf
    return (
        padded.view(mt, 4, 32, ct, 4)
        .transpose(1, 3)
        .reshape(-1)
        .contiguous()
    )


def _unswizzle_reference(swizzled: torch.Tensor, n: int, k: int) -> torch.Tensor:
    scale_cols = k // 32
    mt = (n + 127) // 128
    ct = (scale_cols + 3) // 4
    return (
        swizzled.view(mt, ct, 32, 4, 4)
        .transpose(1, 3)
        .reshape(mt * 128, ct * 4)[:n, :scale_cols]
        .contiguous()
    )


def test_swizzle_round_trip_is_a_lossless_permutation():
    n, k = 8, IN
    scales = _scales((n, k // 32))
    swizzled = _swizzle_reference(scales, n, k)
    # The swizzled block is padded (128-row x 4-col tiles) and reordered...
    assert swizzled.numel() > scales.numel()
    assert not torch.equal(swizzled.reshape(-1)[: scales.numel()], scales.reshape(-1))
    # ...but the round trip through it recovers every scale byte exactly.
    assert torch.equal(_unswizzle_reference(swizzled, n, k), scales)


def test_emulated_mxfp8_gemm_matches_pure_torch_reference():
    # Random data: the pure-torch reference dequantizes the MXFP8 shard
    # and matmuls; the emulated kernel path consumes the RETAINED state
    # through the swizzled scale order. The two must agree exactly.
    torch.manual_seed(0)
    n, k = 8, IN
    payload = _finite_e4m3_bytes((n, k))
    scales = _scales((n, k // 32))
    x = torch.randn(6, k, dtype=torch.bfloat16)
    w_ref = exl3.dequant_minimax_indexer(payload, scales)
    y_ref = torch.nn.functional.linear(x, w_ref)

    retained_fp8, retained_scales = exl3.indexer_mxfp8_retain(payload, scales)
    assert torch.equal(retained_fp8.view(torch.uint8), payload)
    swizzled = _swizzle_reference(retained_scales, n, k)
    recovered = _unswizzle_reference(swizzled, n, k)
    w_emulated = exl3.dequant_minimax_indexer(
        retained_fp8.view(torch.uint8), recovered
    )
    assert torch.equal(w_emulated, w_ref)
    y_emulated = torch.nn.functional.linear(x, w_emulated)
    assert torch.equal(y_emulated, y_ref)
    assert y_emulated.dtype == y_ref.dtype == torch.bfloat16


class _StubExl3Linear:
    """Zero-output stand-in for a LinearEXL3 shard (q/k/v are not under
    test here; the plugin's EXL3 path is untouched by the switch)."""

    def __init__(self, out_features: int):
        self.out_features = out_features

    def forward(self, x, _opts, out_dtype=None, **_kw):
        return torch.zeros(
            x.shape[0], self.out_features, dtype=out_dtype or torch.float32
        )


def _apply_layer(bf16_weight=None, indexer_mxfp8=None):
    layer = SimpleNamespace()
    layer._exl3_linears = [
        _StubExl3Linear(o) for o in OUTS[:3]
    ] + [None, None]
    layer._exl3_linear_output_partition_sizes = list(OUTS)
    layer._exl3_linear_bf16_shards = (
        list(INDEXER_SHARDS) if bf16_weight is not None else []
    )
    layer._exl3_bf16_weight = bf16_weight
    layer._exl3_linear_padded = False
    layer._exl3_indexer_shards = INDEXER_SHARDS
    layer._exl3_indexer_mxfp8 = indexer_mxfp8
    return layer


def test_apply_indexer_slices_match_the_bf16_path_exactly(monkeypatch):
    # Output dtype/shape identical to today's BF16 path, and (because the
    # fake kernel emulates the dequant+matmul semantics) byte-identical
    # outputs for the two indexer slices.
    monkeypatch.delenv("VLLM_EXL3_INDEXER_MXFP8", raising=False)
    payloads, scales = _indexer_stash()
    bf16_weight = torch.cat(
        [exl3.dequant_minimax_indexer(p, s) for p, s in zip(payloads, scales)]
    )
    x = torch.randn(8, IN, dtype=torch.bfloat16)

    layer_bf16 = _apply_layer(bf16_weight=bf16_weight)
    y_bf16 = _method()._apply_impl(layer_bf16, x)
    assert y_bf16.shape == (8, sum(OUTS))
    assert y_bf16.dtype == torch.bfloat16

    kernel = _FakeKernel()
    shims = []
    for p, s, n in zip(payloads, scales, INDEXER_OUTS):
        fp8, sc = exl3.indexer_mxfp8_retain(p, s)
        shims.append((kernel, exl3.make_indexer_mxfp8_shim(fp8, sc)))
    layer_mxfp8 = _apply_layer(indexer_mxfp8=shims)
    y_mxfp8 = _method()._apply_impl(layer_mxfp8, x)
    assert y_mxfp8.shape == y_bf16.shape
    assert y_mxfp8.dtype == y_bf16.dtype
    assert torch.equal(y_mxfp8, y_bf16)
    # The kernel saw exactly the two indexer slices, with the raw bf16
    # activation rows, in shard order.
    assert len(kernel.applied) == 2
    for (shim, xk), n in zip(kernel.applied, INDEXER_OUTS):
        assert xk.dtype == torch.bfloat16 and xk.shape == (8, IN)
        assert tuple(shim.weight.data.shape) == (n, IN)


def test_apply_without_indexer_state_keeps_the_old_routing(monkeypatch):
    # A layer that never carried indexer shards (or a default-off layer
    # whose MXFP8 state was never built) must route exactly as before:
    # no _exl3_indexer_mxfp8 attribute -> no kernel branch at all.
    monkeypatch.delenv("VLLM_EXL3_INDEXER_MXFP8", raising=False)
    payloads, scales = _indexer_stash()
    bf16_weight = torch.cat(
        [exl3.dequant_minimax_indexer(p, s) for p, s in zip(payloads, scales)]
    )
    layer = _apply_layer(bf16_weight=bf16_weight)
    layer._exl3_indexer_mxfp8 = None
    x = torch.randn(4, IN, dtype=torch.bfloat16)
    y = _method()._apply_impl(layer, x)
    assert y.shape == (4, sum(OUTS))
    ref = torch.zeros(4, sum(OUTS))
    ref[:, sum(OUTS[:3]):] = torch.nn.functional.linear(
        x, bf16_weight
    ).to(torch.bfloat16)
    assert torch.equal(y[:, : sum(OUTS[:3])], torch.zeros(4, sum(OUTS[:3]), dtype=torch.bfloat16))
    assert torch.equal(y[:, sum(OUTS[:3]):], ref[:, sum(OUTS[:3]):])


# ---------------------------------------------------------------------------
# (c) memory accounting at the real pack shapes
# ---------------------------------------------------------------------------


def _real_indexer_shapes() -> list[tuple[int, int]]:
    with open(FIXTURE, encoding="utf-8") as fh:
        entries = json.load(fh)
    shapes = []
    for proj in ("index_q_proj", "index_k_proj"):
        w = entries[
            f"language_model.model.layers.0.self_attn.{proj}.weight"
        ]
        s = entries[
            f"language_model.model.layers.0.self_attn.{proj}.weight_scale_inv"
        ]
        assert w["dtype"] == "F8_E4M3"
        assert s["dtype"] == "U8"
        n, k = w["shape"]
        assert s["shape"] == [n, k // 32]
        shapes.append((n, k))
    assert shapes == [(512, 6144), (128, 6144)]
    return shapes


def test_retained_bytes_are_half_of_bf16_at_the_pack_shapes():
    shapes = _real_indexer_shapes()
    bf16_bytes = sum(n * k * 2 for n, k in shapes)
    retained_bytes = sum(n * k + n * (k // 32) for n, k in shapes)
    ratio = retained_bytes / bf16_bytes
    # (1 + 1/32) / 2: the fp8 payload halves the weight and the E8M0
    # block adds 1/32 of it back; strictly between a half and 53%.
    assert 0.5 <= ratio < 0.53
    # Whole-model accounting: 60 indexer layers.
    total_bf16 = bf16_bytes * INDEXER_LAYERS / 2**20
    total_retained = retained_bytes * INDEXER_LAYERS / 2**20
    assert round(total_bf16) == 450  # the live CUDA census's 450 MiB
    assert 225 <= round(total_retained) <= 240
    saving = total_bf16 - total_retained
    assert 210 <= saving <= 226  # ~218 MiB (~0.21 GiB) back to the UMA pool


def test_retained_dtypes_are_fp8_and_uint8_at_the_pack_shapes():
    # Small-scale retention with the pack's aspect: the parameters the
    # switch-on path keeps are fp8 payload + uint8 scales, never BF16.
    shapes = _real_indexer_shapes()
    n, k = shapes[0]
    payload = _finite_e4m3_bytes((n // 32, k // 32))
    scales = _scales((n // 32, k // 32 // 32))
    fp8, sc = exl3.indexer_mxfp8_retain(payload, scales)
    assert fp8.dtype == torch.float8_e4m3fn
    assert fp8.element_size() == 1
    assert sc.dtype == torch.uint8
    assert sc.element_size() == 1
    bf16 = exl3.dequant_minimax_indexer(payload, scales)
    assert bf16.element_size() == 2
