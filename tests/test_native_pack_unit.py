"""CPU-only unit tests for native-pack support: padding, n-gram row geometry,
config validation, codebook marker checks, opaque-op registration, and the
dense linear weight loader's shard-span handling.
"""

import importlib.util
import os

import pytest
import torch

pytest.importorskip("vllm")

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXL3_PATH = os.path.join(_HERE, "..", "src", "vllm_exl3", "exl3.py")

_spec = importlib.util.spec_from_file_location("_exl3_native_pack_unit", _EXL3_PATH)
X = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X)


def test_pad128():
    assert X._exl3_pad128(4304) == 4352
    assert X._exl3_pad128(2560) == 2560


def test_ngram_words_per_row():
    assert X.ngram_words_per_row(5) == 51
    assert X.ngram_words_per_row(3) == 31


def test_ngram_embedding_config_and_lookup():
    cfg = X.Exl3Config(
        bits=3,
        codebook="mul1",
        ngram_embedding={
            "bits": 5,
            "num_shards": 4,
            "rows_per_shard": 1024,
            "num_heads": 2,
            "modules": ["ngram_embedding"],
        },
    )
    spec = cfg._ngram_embedding_spec("model.layers.1.ple.ple_embedding.ngram_embedding")
    assert spec is not None
    assert spec["bits"] == 5

    assert cfg._ngram_embedding_spec("model.layers.1.ple.kv_proj") is None

    with pytest.raises(ValueError):
        X.Exl3Config(bits=3, codebook="mcg", ngram_embedding={"bits": 0})


def test_check_moe_codebook_markers():
    n = 4
    zeros = torch.zeros(n, 1, dtype=torch.int32)
    mul1_all = torch.full((n, 1), X.MUL1_MARKER_SIGNED_INT32, dtype=torch.int32)
    mcg_all = torch.full((n, 1), X.MCG_MARKER_SIGNED_INT32, dtype=torch.int32)

    # All mul1: ok.
    X._check_moe_codebook_markers(zeros, mul1_all, "test")
    # All mcg: ok.
    X._check_moe_codebook_markers(mcg_all, zeros, "test")
    # Neither set: raises.
    with pytest.raises(RuntimeError):
        X._check_moe_codebook_markers(zeros, zeros, "test")
    # Both set: raises.
    with pytest.raises(RuntimeError):
        X._check_moe_codebook_markers(mcg_all, mul1_all, "test")


def test_register_opaque_layer():
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.mlp.gate_proj"
    name = X._exl3_register_opaque_layer(layer, "linear")
    assert name == f"exl3_linear:{layer.prefix}"


def test_dense_loader_span_split():
    cfg = X.Exl3Config(bits=2, codebook="mcg")
    m = X.Exl3LinearMethod(cfg, bits=2)

    layer = torch.nn.Module()
    layer.tp_rank = 0
    layer.tp_size = 1

    loader = m._make_weight_loader("svh", 3, [16, 16, 32], False, [], layer, False)

    param = torch.nn.Parameter(torch.zeros(64, dtype=torch.float16), requires_grad=False)
    loader(param, torch.arange(64, dtype=torch.float16), (0, 1, 2))
    assert torch.equal(param.data, torch.arange(64, dtype=torch.float16))

    param2 = torch.nn.Parameter(torch.zeros(64, dtype=torch.float16), requires_grad=False)
    loader(param2, torch.arange(64, dtype=torch.float16), None)
    assert torch.equal(param2.data, torch.arange(64, dtype=torch.float16))

    bad_param = torch.nn.Parameter(torch.zeros(64, dtype=torch.float16), requires_grad=False)
    with pytest.raises(RuntimeError):
        loader(bad_param, torch.arange(10, dtype=torch.float16), (0, 1))
