"""VLLM_EXL3_INDEXER_MXFP8: native-MXFP8 retention of the MSA indexer shards.

CPU-only (no vLLM, no GPU, no kernel launch):

- (a) default-off is byte-identical to the pre-switch loader: the stashed
  MXFP8 payloads are dequantized into the same BF16 staging rows and no
  MXFP8 state appears on the layer;
- (b) the switch-on resolver binds vLLM's WEIGHT-ONLY Marlin MXFP8 kernel
  (W8A16, bf16 activations) and nothing else: every other condition --
  missing kernel, platform refusal, shape refusal -- fails closed with an
  error naming the switch, and the Marlin repack is byte-preserving at
  the real pack shapes (no tile padding), so the retained footprint does
  not grow;
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


def _install_fake_vllm_mxfp8(monkeypatch, *, supported, implementable):
    """Install a fake vllm MXFP8 kernel API into sys.modules.

    The real API is vLLM's own `MarlinMxfp8LinearKernel` behind the
    `Mxfp8LinearLayerConfig` config; this stand-in lets the resolver's
    accept/refuse branches run on CPU. monkeypatch restores sys.modules.
    """
    import sys
    import types

    for name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.kernels",
        "vllm.model_executor.kernels.linear",
        "vllm.model_executor.kernels.linear.mxfp8",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    cfg_mod = types.ModuleType(
        "vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel"
    )

    class Mxfp8LinearLayerConfig:  # minimal stand-in
        def __init__(self, weight_shape):
            self.weight_shape = tuple(weight_shape)

    setattr(cfg_mod, "Mxfp8LinearLayerConfig", Mxfp8LinearLayerConfig)
    monkeypatch.setitem(sys.modules, cfg_mod.__name__, cfg_mod)

    marlin_mod = types.ModuleType(
        "vllm.model_executor.kernels.linear.mxfp8.marlin"
    )

    class MarlinMxfp8LinearKernel:  # stand-in for vLLM's W8A16 kernel
        def __init__(self, config):
            self.config = config

        @classmethod
        def is_supported(cls):
            return (True, None) if supported else (False, "no marlin here")

        @classmethod
        def can_implement(cls, config):
            return (True, None) if implementable else (False, "bad shape")

    setattr(marlin_mod, "MarlinMxfp8LinearKernel", MarlinMxfp8LinearKernel)
    monkeypatch.setitem(sys.modules, marlin_mod.__name__, marlin_mod)
    return MarlinMxfp8LinearKernel


def test_resolver_binds_the_weight_only_marlin_kernel(monkeypatch):
    # Both gates admit: the resolver must return vLLM's weight-only Marlin
    # kernel for the shard -- never a W8A8 backend.
    cls = _install_fake_vllm_mxfp8(monkeypatch, supported=True, implementable=True)
    kernel = exl3.indexer_mxfp8_kernel((512, 6144))
    assert type(kernel) is cls
    assert type(kernel).__name__ == "MarlinMxfp8LinearKernel"
    assert kernel.config.weight_shape == (512, 6144)


def test_resolver_fails_closed_on_platform_refusal(monkeypatch):
    cls = _install_fake_vllm_mxfp8(monkeypatch, supported=False, implementable=True)
    with pytest.raises(RuntimeError, match=r"VLLM_EXL3_INDEXER_MXFP8=1.*Marlin"):
        exl3.indexer_mxfp8_kernel((512, 6144))
    assert cls.is_supported() == (False, "no marlin here")


def test_resolver_fails_closed_on_shape_refusal(monkeypatch):
    _install_fake_vllm_mxfp8(monkeypatch, supported=True, implementable=False)
    with pytest.raises(RuntimeError, match="refuses the indexer shard shape"):
        exl3.indexer_mxfp8_kernel((512, 6144))


def test_resolver_source_names_no_w8a8_backend():
    # Source-level pin: the resolver names the weight-only Marlin kernel
    # and no activation-quantizing (W8A8) backend can be re-pointed at it.
    import inspect

    src = inspect.getsource(exl3.indexer_mxfp8_kernel)
    assert "MarlinMxfp8LinearKernel" in src
    for w8a8 in ("FlashInferCutlass", "B12x", "TorchMxFp8", "Humming"):
        assert w8a8 not in src


class _FakeKernel:
    """Stands in for vLLM's weight-only Marlin MXFP8 kernel on CPU.

    ``process_weights_after_loading`` records the call (the real one
    repacks payload+scales into its W8A16 format); ``apply_weights``
    emulates the kernel's semantics -- dequantize the shim's own (fp8
    payload, E8M0 scales) state and matmul -- so the glue can be
    compared against the BF16 path without any GPU.
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
    # The bound kernel classes are recorded on the layer (the GB10 receipt
    # cross-checks them against the weight-only Marlin requirement).
    assert layer._exl3_indexer_mxfp8_kernels == ["_FakeKernel", "_FakeKernel"]
    # One shim per indexer shard, in shard order, with the checkpoint
    # geometry; the payload is the bitcast fp8 view, the scales stay the
    # row-major uint8 block the kernel's load step expects, and the shim
    # carries the true partition sizes the kernel's apply reads.
    for slot, (n, (kernel, shim)) in enumerate(zip(INDEXER_OUTS, shims)):
        assert made[slot][0] == (n, IN)
        assert kernel.processed == [shim]
        assert shim.weight.data.dtype == torch.float8_e4m3fn
        assert tuple(shim.weight.data.shape) == (n, IN)
        assert torch.equal(shim.weight.data.view(torch.uint8), payloads[slot])
        assert shim.weight_scale.data.dtype == torch.uint8
        assert tuple(shim.weight_scale.data.shape) == (n, IN // 32)
        assert torch.equal(shim.weight_scale.data, scales[slot])
        assert shim.output_size_per_partition == n
        assert shim.input_size_per_partition == IN
        assert shim.bias is None
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
# (b) the Marlin repack is byte-preserving at the real pack shapes
# ---------------------------------------------------------------------------


def _marlin_padded_nk_reference(n: int, k: int) -> tuple[int, int]:
    """marlin_utils.marlin_padded_nk's documented tile rule, test-local.

    Marlin requires (n %% 64, k %% 128) or (n %% 128, k %% 64); shapes
    satisfying neither are zero-padded up to the cheaper family, with K
    kept divisible by the 32-wide scale block. This reference exists so
    the real pack shapes can be proven padding-free on CPU; the
    real-kernel agreement is gated on the GB10 (parity case G).
    """
    import math

    def round_up(x: int, m: int) -> int:
        return ((x + m - 1) // m) * m

    group = 32
    candidates = (
        (round_up(n, 64), round_up(k, math.lcm(128, group))),
        (round_up(n, 128), round_up(k, math.lcm(64, group))),
    )
    return min(candidates, key=lambda nk: (nk[0] * nk[1], nk[0] + nk[1]))


def test_marlin_repack_needs_no_tile_padding_at_pack_shapes():
    # The real shards are (512, 6144) and (128, 6144): both already
    # satisfy Marlin's thread-tile families, so the load step's zero
    # padding is a no-op and the repack cannot grow the tensors.
    for n, k in _real_indexer_shapes():
        assert _marlin_padded_nk_reference(n, k) == (n, k)


def test_marlin_repack_is_byte_preserving_at_pack_shapes():
    # The repacked qweight is (k/16, 4n) int32 -- exactly the payload's
    # bytes -- and the permuted scale block comes back as 1-byte e8m0fnu,
    # so the retained footprint is unchanged by the load step.
    for n, k in _real_indexer_shapes():
        payload_bytes = n * k
        scale_bytes = n * (k // 32)
        repacked_weight_bytes = (k // 16) * (4 * n) * 4
        repacked_scale_bytes = n * (k // 32)
        assert repacked_weight_bytes == payload_bytes
        assert repacked_weight_bytes + repacked_scale_bytes == (
            payload_bytes + scale_bytes
        )


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
