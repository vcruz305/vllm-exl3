"""CPU regression tests for EXL3 expert-map and TP loader geometry."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import vllm_exl3.exl3 as exl3


def _new_moe_method(moe, cfg, bits: int = 4):
    """Construct Exl3MoEMethod without requiring a real vLLM FusedMoEMethodBase."""
    method = object.__new__(exl3.Exl3MoEMethod)
    method.moe = moe
    method.quant_config = cfg
    method.bits = int(bits)
    method._logged = False
    return method


class _ReadOnlyExpertMapLayer:
    def __init__(self, expert_map: torch.Tensor) -> None:
        self._raw_expert_map = expert_map

    @property
    def expert_map(self) -> torch.Tensor:
        return self._raw_expert_map


class _MoEOwner(torch.nn.Module):
    tp_rank = 0
    tp_size = 1
    moe_tp_size = 1

    def __init__(self) -> None:
        super().__init__()
        self.layer_name = "layers.0.ffn.experts"
        self.global_num_experts = 1
        self.local_num_experts = 1
        self.starting_expert_offset = 0
        self.moe_config = SimpleNamespace(
            hidden_dim=16,
            num_experts=1,
            num_local_experts=1,
            experts_per_token=1,
            activation="silu",
            rocm_aiter_fmoe_enabled=False,
            swiglu_limit=None,
        )

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        return expert_id if 0 <= int(expert_id) < 1 else -1


def test_moe_loader_prefers_layer_tp_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A MoE layer with TP=1 must not inherit process TP=8 slicing."""
    monkeypatch.setenv("VLLM_EXL3_TRELLIS_ARENA", "0")
    monkeypatch.setenv("VLLM_EXL3_ARENA_PRESCAN", "0")
    owner = _MoEOwner()
    method = _new_moe_method(owner.moe_config, exl3.Exl3Config(bits=2, codebook="mcg", scope="test"), bits=2)
    method.create_weights(
        owner,
        num_experts=1,
        hidden_size=16,
        intermediate_size_per_partition=16,
        params_dtype=torch.bfloat16,
    )
    owner.moe_tp_size = 1
    owner.tp_rank = 0
    loaded = torch.arange(1 * 1 * 32, dtype=torch.int16).reshape(1, 1, 32)
    ok = method._load_exl3(
        owner.w13_trellis,
        loaded,
        "experts.w13_trellis",
        shard_id="w1",
        expert_id=0,
        return_success=True,
    )
    assert ok is True
    torch.testing.assert_close(owner.gate_trellis[0], loaded)


def test_pin_expert_map_uses_private_cache_for_read_only_property() -> None:
    raw = torch.tensor([2, 0, 1], dtype=torch.int32)
    layer = _ReadOnlyExpertMapLayer(raw)

    pinned = exl3.pin_exl3_expert_map(layer, torch.device("cpu"))

    assert pinned is not None
    assert pinned.dtype == torch.long
    assert pinned.device == torch.device("cpu")
    assert layer._exl3_pinned_expert_map is pinned
    assert exl3.pin_exl3_expert_map(layer, torch.device("cpu")) is pinned

    layer._raw_expert_map = torch.tensor([1, 2, 0], dtype=torch.int64)
    refreshed = exl3.pin_exl3_expert_map(layer, torch.device("cpu"))
    assert refreshed is not pinned
    torch.testing.assert_close(refreshed, layer._raw_expert_map)


@pytest.mark.parametrize(
    ("is_row_parallel", "tp_rank", "expected"),
    [
        (True, 1, torch.arange(256, dtype=torch.float32).reshape(16, 16)[:, 8:]),
        (False, 1, torch.arange(256, dtype=torch.float32).reshape(16, 16)[8:, :]),
    ],
)
def test_dense_bf16_loader_slices_correct_tp_axis(
    is_row_parallel: bool,
    tp_rank: int,
    expected: torch.Tensor,
) -> None:
    method = object.__new__(exl3.Exl3LinearMethod)
    layer = SimpleNamespace(tp_rank=tp_rank, tp_size=2)
    param_shape = (16, 8) if is_row_parallel else (8, 16)
    param = torch.nn.Parameter(torch.zeros(param_shape), requires_grad=False)
    output_sizes = [16] if is_row_parallel else [8]
    loader = method._make_weight_loader(
        "weight", 1, output_sizes, is_row_parallel, [0], layer, False
    )
    loaded = torch.arange(256, dtype=torch.float32).reshape(16, 16)

    loader(param, loaded)

    torch.testing.assert_close(param, expected)


def test_qkv_loader_replicated_kv_heads_uses_shard_specific_tp() -> None:
    method = object.__new__(exl3.Exl3LinearMethod)
    layer = SimpleNamespace(tp_rank=7, tp_size=8)
    output_sizes = [512, 128, 128]
    total_tiles = sum(output_sizes) // 16
    param = torch.nn.Parameter(
        torch.zeros(1, total_tiles, 32, dtype=torch.int16), requires_grad=False
    )
    loader = method._make_weight_loader(
        "trellis", 3, output_sizes, False, [], layer, True
    )
    loaded = torch.arange(1 * (128 // 16) * 32, dtype=torch.int16).reshape(
        1, 128 // 16, 32
    )

    loader(param, loaded, loaded_shard_id="k")

    torch.testing.assert_close(param[:, 512 // 16 : (512 + 128) // 16], loaded)

    svh = torch.nn.Parameter(
        torch.zeros(sum(output_sizes), dtype=torch.float16), requires_grad=False
    )
    svh_loader = method._make_weight_loader(
        "svh", 3, output_sizes, False, [], layer, True
    )
    loaded_svh = torch.arange(128, dtype=torch.float16)
    svh_loader(svh, loaded_svh, loaded_shard_id="k")
    torch.testing.assert_close(svh[512:640], loaded_svh)


def test_suh_shape_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scale tensors still hard-fail on shape mismatch (exact dest geometry)."""
    monkeypatch.setenv("VLLM_EXL3_TRELLIS_ARENA", "0")
    owner = _MoEOwner()
    method = _new_moe_method(owner.moe_config, exl3.Exl3Config(bits=2, codebook="mcg", scope="test"), bits=2)
    method.create_weights(
        owner,
        num_experts=1,
        hidden_size=16,
        intermediate_size_per_partition=16,
        params_dtype=torch.bfloat16,
    )
    owner.moe_tp_size = 1
    owner.tp_rank = 0
    with pytest.raises(RuntimeError, match="shape mismatch"):
        method._load_exl3(
            owner.w13_suh,
            torch.arange(8, dtype=torch.float16),
            "experts.w13_suh",
            shard_id="w1",
            expert_id=0,
            return_success=True,
        )


def test_trellis_exact_shape_replaces_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_EXL3_TRELLIS_ARENA", "0")
    owner = _MoEOwner()
    method = _new_moe_method(owner.moe_config, exl3.Exl3Config(bits=2, codebook="mcg", scope="test"), bits=2)
    method.create_weights(
        owner,
        num_experts=1,
        hidden_size=16,
        intermediate_size_per_partition=16,
        params_dtype=torch.bfloat16,
    )
    owner.moe_tp_size = 1
    owner.tp_rank = 0
    loaded = torch.arange(1 * 1 * 48, dtype=torch.int16).reshape(1, 1, 48)  # K=3
    assert method._load_exl3(
        owner.w13_trellis,
        loaded,
        "experts.0.w1.trellis",
        shard_id="w1",
        expert_id=0,
        return_success=True,
    )
    assert tuple(owner.gate_trellis[0].shape) == (1, 1, 48)
    torch.testing.assert_close(owner.gate_trellis[0], loaded)
