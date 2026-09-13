"""Direct-to-arena fill vs legacy on synthetic mixed-K tensors."""

from __future__ import annotations

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


class _MoeCfg:
    hidden_dim = 64
    num_experts = 4
    num_local_experts = 4
    experts_per_token = 2
    activation = "silu"
    rocm_aiter_fmoe_enabled = False
    swiglu_limit = None


class _Layer(torch.nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.moe_config = _MoeCfg()
        self.layer_name = "layers.0.ffn.experts"
        self.global_num_experts = n
        self.local_num_experts = n
        self.starting_expert_offset = 0

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        return int(expert_id) if 0 <= int(expert_id) < self.local_num_experts else -1


def _stub_linear(trellis, suh, svh, mcg=None, mul1=None, **_kwargs):
    return SimpleNamespace(
        trellis=trellis, suh=suh, svh=svh, mcg=True, mul1=False,
        forward=lambda *a, **k: None,
    )


def _make(n: int = 4):
    method = _new_moe_method(_MoeCfg(), exl3.Exl3Config(bits=4, codebook="mcg", scope="test"), bits=4)
    layer = _Layer(n)
    method.create_weights(
        layer,
        num_experts=n,
        hidden_size=64,
        intermediate_size_per_partition=64,
        params_dtype=torch.bfloat16,
    )
    layer.moe_tp_size = 1
    layer.tp_rank = 0
    return method, layer


def _trellis(in_t, out_t, k):
    return torch.arange(in_t * out_t * k * 16, dtype=torch.int16).reshape(
        in_t, out_t, k * 16
    )


def _load_all(method, layer, ks_gate, ks_up, ks_down, *, arena: bool):
    import os

    os.environ["VLLM_EXL3_TRELLIS_ARENA"] = "1" if arena else "0"
    os.environ["VLLM_EXL3_ARENA_PRESCAN"] = "0"
    for eid, (kg, ku, kd) in enumerate(zip(ks_gate, ks_up, ks_down)):
        payloads = {
            ("w1", "trellis"): _trellis(4, 4, kg),
            ("w3", "trellis"): _trellis(4, 4, ku),
            ("w2", "trellis"): _trellis(4, 4, kd),
            ("w1", "suh"): torch.ones(64, dtype=torch.float16),
            ("w3", "suh"): torch.ones(64, dtype=torch.float16),
            ("w2", "suh"): torch.ones(64, dtype=torch.float16),
            ("w1", "svh"): torch.ones(64, dtype=torch.float16),
            ("w3", "svh"): torch.ones(64, dtype=torch.float16),
            ("w2", "svh"): torch.ones(64, dtype=torch.float16),
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
            assert method._load_exl3(
                param,
                tensor,
                f"experts.{eid}.{proj}.{kind}",
                shard_id=proj,
                expert_id=eid,
                return_success=True,
            )


def test_direct_vs_legacy_parity_and_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exl3, "make_linear_exl3", _stub_linear)
    ks = [3, 8, 4, 5]
    ups = [4, 8, 4, 5]  # expert 0: w1=K3, w3=K4

    method_l, layer_l = _make(4)
    _load_all(method_l, layer_l, ks, ups, ks, arena=False)
    method_l.process_weights_after_loading(layer_l)

    method_d, layer_d = _make(4)
    monkeypatch.setenv("VLLM_EXL3_TRELLIS_ARENA", "1")
    monkeypatch.setenv("VLLM_EXL3_ARENA_PRESCAN", "0")
    shapes = {
        "gate": {i: (4, 4, k * 16) for i, k in enumerate(ks)},
        "up": {i: (4, 4, k * 16) for i, k in enumerate(ups)},
        "down": {i: (4, 4, k * 16) for i, k in enumerate(ks)},
    }
    exl3.prepare_trellis_arena_plan(layer_d, shapes)
    _load_all(method_d, layer_d, ks, ups, ks, arena=True)
    method_d.process_weights_after_loading(layer_d)

    assert layer_d._exl3_mixed_k is True
    assert exl3._count_trellis_storages(layer_d) < exl3._count_trellis_storages(layer_l)
    assert exl3._trellis_nbytes(layer_d) == exl3._trellis_nbytes(layer_l)
    assert layer_d._exl3_trellis_arena_stats.get("mode") == "direct_plan"
    assert (
        int(layer_d._exl3_trellis_arena_stats["temp_peak_bytes"])
        < int(layer_d._exl3_trellis_arena_stats["final_bytes"]) // 4
    )
    for eid in range(4):
        assert torch.equal(layer_d.gate_trellis[eid].cpu(), layer_l.gate_trellis[eid].cpu())
        assert torch.equal(layer_d.up_trellis[eid].cpu(), layer_l.up_trellis[eid].cpu())
        assert torch.equal(layer_d.down_trellis[eid].cpu(), layer_l.down_trellis[eid].cpu())
    assert layer_d.gate_trellis[0].shape[-1] // 16 == 3
    assert layer_d.up_trellis[0].shape[-1] // 16 == 4
