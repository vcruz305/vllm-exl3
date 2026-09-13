"""Synthetic CPU tests for per-expert heterogeneous packed-K EXL3 MoE loads."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import vllm_exl3
import vllm_exl3.exl3 as exl3


def _new_moe_method(moe, cfg, bits: int = 4):
    """Construct Exl3MoEMethod without requiring a real vLLM FusedMoEMethodBase."""
    method = object.__new__(exl3.Exl3MoEMethod)
    method.moe = moe
    method.quant_config = cfg
    method.bits = int(bits)
    method._logged = False
    return method


class _MoeCfg:
    hidden_dim = 64
    num_experts = 4
    num_local_experts = 4
    experts_per_token = 2
    activation = "silu"
    rocm_aiter_fmoe_enabled = False
    swiglu_limit = None


class _Layer(torch.nn.Module):
    def __init__(self, n_experts: int) -> None:
        super().__init__()
        self.moe_config = _MoeCfg()
        self.layer_name = "layers.0.ffn.experts"
        self.global_num_experts = n_experts
        self.local_num_experts = n_experts
        self.starting_expert_offset = 0

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        return int(expert_id) if 0 <= int(expert_id) < self.local_num_experts else -1


def _stub_linear(trellis, suh, svh, mcg=None, mul1=None, **_kwargs):
    return SimpleNamespace(
        trellis=trellis,
        suh=suh,
        svh=svh,
        mcg=True if mcg is None else bool(int(mcg.reshape(-1)[0].item()) != 0)
        if mcg.numel()
        else True,
        mul1=False if mul1 is None else bool(int(mul1.reshape(-1)[0].item()) != 0)
        if mul1.numel()
        else False,
        forward=lambda *a, **k: None,
    )


def _make_method_layer(n_experts: int = 4, bits: int = 4, hidden: int = 64, inter: int = 64):
    cfg = exl3.Exl3Config(bits=bits, codebook="mcg", scope="test")
    method = _new_moe_method(_MoeCfg(), cfg, bits=bits)
    layer = _Layer(n_experts)
    method.create_weights(
        layer,
        num_experts=n_experts,
        hidden_size=hidden,
        intermediate_size_per_partition=inter,
        params_dtype=torch.bfloat16,
    )
    layer.moe_tp_size = 1
    layer.tp_rank = 0
    return method, layer


def _trellis(in_tiles: int, out_tiles: int, k: int) -> torch.Tensor:
    return torch.arange(in_tiles * out_tiles * k * 16, dtype=torch.int16).reshape(
        in_tiles, out_tiles, k * 16
    )


def _load_expert(method, layer, eid: int, k_gate: int, k_up: int, k_down: int) -> None:
    in_tiles = layer._exl3_in_tiles
    out_tiles = layer._exl3_out_tiles
    hidden = layer._exl3_hidden_size
    inter = layer._exl3_intermediate_local
    payloads = {
        ("w1", "trellis"): _trellis(in_tiles, out_tiles, k_gate),
        ("w3", "trellis"): _trellis(in_tiles, out_tiles, k_up),
        ("w2", "trellis"): _trellis(out_tiles, in_tiles, k_down),
        ("w1", "suh"): torch.ones(hidden, dtype=torch.float16),
        ("w3", "suh"): torch.ones(hidden, dtype=torch.float16),
        ("w2", "suh"): torch.ones(inter, dtype=torch.float16),
        ("w1", "svh"): torch.ones(inter, dtype=torch.float16),
        ("w3", "svh"): torch.ones(inter, dtype=torch.float16),
        ("w2", "svh"): torch.ones(hidden, dtype=torch.float16),
        ("w1", "mcg"): torch.tensor(exl3.MCG_MARKER_SIGNED_INT32, dtype=torch.int32),
        ("w3", "mcg"): torch.tensor(exl3.MCG_MARKER_SIGNED_INT32, dtype=torch.int32),
        ("w2", "mcg"): torch.tensor(exl3.MCG_MARKER_SIGNED_INT32, dtype=torch.int32),
    }
    for (proj, kind), tensor in payloads.items():
        if kind == "trellis":
            param = layer.w13_trellis if proj in ("w1", "w3") else layer.w2_trellis
        else:
            param = (
                getattr(layer, f"w13_{kind}")
                if proj in ("w1", "w3")
                else getattr(layer, f"w2_{kind}")
            )
        ok = method._load_exl3(
            param,
            tensor,
            f"layers.0.ffn.experts.{eid}.{proj}.{kind}",
            shard_id=proj,
            expert_id=eid,
            return_success=True,
        )
        assert ok is True, (eid, proj, kind)


@pytest.fixture(autouse=True)
def _legacy_arena_default(monkeypatch: pytest.MonkeyPatch):
    # Default unit path: legacy per-expert allocs unless a test opts into arenas.
    monkeypatch.setenv("VLLM_EXL3_TRELLIS_ARENA", "0")
    monkeypatch.setenv("VLLM_EXL3_ARENA_PRESCAN", "0")
    monkeypatch.setenv("EXL3_FUSED_MOE", "0")


def test_create_weights_uses_ragged_parameter_lists() -> None:
    method, layer = _make_method_layer(n_experts=3)
    assert len(layer.gate_trellis) == 3
    assert len(layer.up_trellis) == 3
    assert len(layer.down_trellis) == 3
    assert tuple(layer.w13_trellis.shape) == (0,)
    assert getattr(layer, "_exl3_mixed_k", None) is False


def test_exact_shape_load_for_k_classes_3_through_8(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exl3, "make_linear_exl3", _stub_linear)
    method, layer = _make_method_layer(n_experts=6, bits=4)
    # One expert per K in 3..8; also force w1!=w3 on expert 0.
    k_classes = list(range(3, 9))
    for eid, k in enumerate(k_classes):
        k_gate = k
        k_up = 4 if eid == 0 else k  # expert 0: w1=K3, w3=K4
        k_down = k
        _load_expert(method, layer, eid, k_gate, k_up, k_down)

    method.process_weights_after_loading(layer)
    assert layer._exl3_mixed_k is True
    assert getattr(layer, "_exl3_ptrs", None) in (None, {})
    assert len(layer._exl3_inners) == 6
    assert tuple(layer.gate_trellis[0].shape)[-1] // 16 == 3
    assert tuple(layer.up_trellis[0].shape)[-1] // 16 == 4
    for eid, k in enumerate(k_classes):
        assert tuple(layer.gate_trellis[eid].shape)[-1] // 16 == (3 if eid == 0 else k)
        assert tuple(layer.down_trellis[eid].shape)[-1] // 16 == k


def test_intra_expert_w1_ne_w3_forces_python_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exl3, "make_linear_exl3", _stub_linear)
    method, layer = _make_method_layer(n_experts=2, bits=4)
    _load_expert(method, layer, 0, k_gate=3, k_up=5, k_down=4)
    _load_expert(method, layer, 1, k_gate=3, k_up=5, k_down=4)
    method.process_weights_after_loading(layer)
    assert layer._exl3_mixed_k is True
    assert layer._exl3_ptrs is None


def test_uniform_k_keeps_mixed_flag_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exl3, "make_linear_exl3", _stub_linear)
    method, layer = _make_method_layer(n_experts=2, bits=4)
    _load_expert(method, layer, 0, 4, 4, 4)
    _load_expert(method, layer, 1, 4, 4, 4)
    method.process_weights_after_loading(layer)
    assert layer._exl3_mixed_k is False
    # No fused kernel in this CPU env, but mixed_k must not be the reason.
    assert getattr(layer, "_exl3_ptrs", None) in (None, {})


def test_trellis_tile_mismatch_raises() -> None:
    method, layer = _make_method_layer(n_experts=1)
    bad = torch.zeros(2, 2, 64, dtype=torch.int16)  # wrong tile prefix
    with pytest.raises(RuntimeError, match="tile mismatch"):
        method._load_exl3(
            layer.w13_trellis,
            bad,
            "layers.0.ffn.experts.0.w1.trellis",
            shard_id="w1",
            expert_id=0,
            return_success=True,
        )


def test_runtime_diagnostics_advertise_tensor_level_mixed_k() -> None:
    diag = vllm_exl3.runtime_diagnostics()
    mixed = diag["mixed_k"]
    assert mixed["tensor_level_mixed_k_within_layer"] is True
    assert mixed["heterogeneous_dispatch"] == "python_loop"
    assert "python_loop" in mixed["note"]
