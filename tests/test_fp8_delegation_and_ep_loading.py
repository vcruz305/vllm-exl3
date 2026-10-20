"""Focused CPU regression tests for FP8 delegation propagation and EP expert loading.

Covers the four behaviours the DSV4.1 TP4+EP4 bring-up depends on:

1. A pack that keeps non-routed (dense) weights in source block-FP8 declares the
   delegate under the *nested* ``quantization_config.non_routed_quantization``
   mapping; the outer EXL3 config must still surface its ``weight_block_size``
   as a list so architecture-level probes compare equal with the source
   ``[32, 32]``.
2. EP keeps whole experts: ``RoutedExperts`` reporting ``use_ep`` must not have
   ``shard_exl3_col`` / ``shard_exl3_row`` quartered feature slicing applied.
3. The wrong-rank expert filter (global -> local mapping returning ``-1``) stays
   authoritative on that path: no write happens for experts this rank does not
   own.
4. EXL3 is never treated as FP8: an EXL3-only pack reports no block shape, and
   the block-shape property is single-sourced from the V4.1 compat shim.

No vLLM runtime, GPU, model config or checkpoint is required: the MoE method is
constructed directly with the same ``object.__new__`` harness used by
``tests/test_loader_parity.py``.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import vllm_exl3.exl3 as exl3
from vllm_exl3.deepseek_v41 import (
    install_deepseek_v41_compat,
    is_deepseek_v41_source_quant,
    source_weight_block_size,
)

# The delegate block declared by DSV4.1-Flash-EXL3 packs that keep the dense
# (non-routed) weights in the source block-FP8 format. This mapping is nested
# under the pack's outer ``quantization_config``.
PACK_NON_ROUTED_QUANT = {
    "quant_method": "deepseek_v4_fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
}


def _pack_config(**overrides):
    values = {
        "quant_method": "exl3",
        "bits": 4,
        "codebook": "mcg",
        "scope": "glm53_routed_experts_only",
        "non_routed_quantization": dict(PACK_NON_ROUTED_QUANT),
    }
    values.update(overrides)
    return values


def _v41_config_cls():
    """Exl3Config with the V4.1 compat shim installed, as ``register()`` does."""
    install_deepseek_v41_compat(exl3)
    return exl3.Exl3Config


# --------------------------------------------------------------------------- #
# 1. runtime-delegated FP8 propagation through a nested config
# --------------------------------------------------------------------------- #


def test_nested_non_routed_fp8_delegate_surfaces_block_shape():
    config_cls = _v41_config_cls()
    config = config_cls.from_config(_pack_config())

    # The delegate mapping is stored verbatim (it is resolved to a real quant
    # method at runtime by ``_non_routed_delegate``), yet the outer config must
    # already answer architecture probes.
    assert config.non_routed_quantization == PACK_NON_ROUTED_QUANT
    assert config.weight_block_size == [32, 32]
    assert isinstance(config.weight_block_size, list)  # a tuple compares unequal
    assert config.weight_block_size == config.non_routed_quantization["weight_block_size"]
    assert config.has_blocked_weights() is True
    assert is_deepseek_v41_source_quant(config) is True


def test_delegate_block_shape_is_surfaced_without_a_live_delegate_resolver():
    """The probe must not depend on vLLM resolving the delegate first."""
    config_cls = _v41_config_cls()
    config = config_cls.from_config(
        _pack_config(
            non_routed_quantization={
                "quant_method": "deepseek_v4_fp8",
                "weight_block_size": [32, 32],
            }
        )
    )
    assert getattr(config, "_nr_delegate_cached", None) is None
    assert config.weight_block_size == [32, 32]
    assert source_weight_block_size(config) == [32, 32]


def test_delegate_block_shape_accepts_a_tuple_source_pair():
    config_cls = _v41_config_cls()
    config = config_cls.from_config(
        _pack_config(
            non_routed_quantization={
                "quant_method": "deepseek_v4_fp8",
                "weight_block_size": (32, 32),
            }
        )
    )
    assert config.weight_block_size == [32, 32]
    assert isinstance(config.weight_block_size, list)


def test_delegate_block_shape_rejects_malformed_pairs():
    config_cls = _v41_config_cls()
    for declared in ([32], [32, 32, 32], [0, 32], "32", None):
        config = config_cls.from_config(
            _pack_config(
                non_routed_quantization={
                    "quant_method": "deepseek_v4_fp8",
                    "weight_block_size": declared,
                }
            )
        )
        assert config.weight_block_size is None, declared


# --------------------------------------------------------------------------- #
# 2. EXL3 vs FP8 separation on the outer config
# --------------------------------------------------------------------------- #


def test_exl3_only_pack_is_not_reported_as_fp8():
    config_cls = _v41_config_cls()
    config = config_cls.from_config(
        _pack_config(non_routed_quantization=None)
    )

    assert config.get_name() == "exl3"
    assert config.bits == 4
    assert config.weight_block_size is None
    assert config.has_blocked_weights() is False
    assert is_deepseek_v41_source_quant(config) is False


def test_block_size_property_is_single_sourced_from_the_v41_shim():
    """No duplicate class-body override: the shim owns the property."""
    config_cls = _v41_config_cls()
    install_deepseek_v41_compat(exl3)  # idempotent
    assert config_cls.weight_block_size.fget is source_weight_block_size


# --------------------------------------------------------------------------- #
# 3/4. EP whole-expert loading, TP slicing, wrong-rank filter
# --------------------------------------------------------------------------- #

HIDDEN = 16
WHOLE_EXPERTS = 4  # global MoE TP world that used to quarter whole experts


def _new_moe_method(moe, cfg, bits: int = 2):
    """Exl3MoEMethod without requiring a real vLLM FusedMoEMethodBase."""
    method = object.__new__(exl3.Exl3MoEMethod)
    method.moe = moe
    method.quant_config = cfg
    method.bits = int(bits)
    method._logged = False
    return method


class _MoEOwner(torch.nn.Module):
    """RoutedExperts stand-in holding whole experts under EP."""

    tp_rank = 0
    tp_size = 1
    moe_tp_size = 1

    def __init__(self, *, use_ep: bool, tp_size: int) -> None:
        super().__init__()
        self.use_ep = bool(use_ep)
        self.tp_rank = 0
        self.tp_size = int(tp_size)
        self.moe_tp_size = int(tp_size)
        self.layer_name = "layers.0.ffn.experts"
        self.local_num_experts = 1
        self.global_num_experts = 1 * int(tp_size)
        self.starting_expert_offset = 0
        self.moe_config = SimpleNamespace(
            hidden_dim=HIDDEN,
            num_experts=self.global_num_experts,
            num_local_experts=self.local_num_experts,
            experts_per_token=1,
            activation="silu",
            rocm_aiter_fmoe_enabled=False,
            swiglu_limit=None,
        )

    # Authoritative ownership filter: -1 means "this rank does not own it".
    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        local = int(expert_id) - int(self.starting_expert_offset)
        return local if 0 <= local < self.local_num_experts else -1


def _moe_setup(monkeypatch, *, use_ep: bool, intermediate: int, tp_size: int):
    monkeypatch.setenv("VLLM_EXL3_TRELLIS_ARENA", "0")
    monkeypatch.setenv("VLLM_EXL3_ARENA_PRESCAN", "0")
    owner = _MoEOwner(use_ep=use_ep, tp_size=tp_size)
    method = _new_moe_method(
        owner.moe_config,
        exl3.Exl3Config(bits=2, codebook="mcg", scope="test"),
        bits=2,
    )
    method.create_weights(
        owner,
        num_experts=owner.local_num_experts,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=intermediate,
        params_dtype=torch.bfloat16,
    )
    return owner, method


def test_use_ep_keeps_whole_expert_scale_rows_unsliced(monkeypatch):
    """EP: svh holds the whole 5120x2304 expert, so no feature slicing.

    Under the aligned scheme the per-rank svh buffer is padded to the
    128-channel block boundary; the expert window lands at the head and the
    pad stays zero."""
    whole_intermediate = HIDDEN * WHOLE_EXPERTS
    owner, method = _moe_setup(
        monkeypatch, use_ep=True, intermediate=whole_intermediate, tp_size=WHOLE_EXPERTS
    )
    whole_svh = torch.arange(whole_intermediate, dtype=torch.float16)

    ok = method._load_exl3(
        owner.w13_svh,
        whole_svh,
        "experts.w13_svh",
        shard_id="w1",
        expert_id=0,
        return_success=True,
    )

    assert ok is True
    row = owner.w13_svh.data[0, 0]
    torch.testing.assert_close(row[:whole_intermediate], whole_svh)
    assert bool((row[whole_intermediate:] == 0).all())


def test_use_ep_keeps_whole_expert_suh_rows_unsliced(monkeypatch):
    """EP: w2 svh/suh carry whole-expert geometry for the row-parallel leg."""
    whole_intermediate = HIDDEN * WHOLE_EXPERTS
    owner, method = _moe_setup(
        monkeypatch, use_ep=True, intermediate=whole_intermediate, tp_size=WHOLE_EXPERTS
    )
    whole_svh = torch.arange(HIDDEN, dtype=torch.float16)

    ok = method._load_exl3(
        owner.w2_svh,
        whole_svh,
        "experts.w2_svh",
        shard_id="w2",
        expert_id=0,
        return_success=True,
    )

    assert ok is True
    torch.testing.assert_close(owner.w2_svh.data[0], whole_svh)


def test_whole_expert_scales_without_ep_guard_reproduce_the_pr16_mismatch(
    monkeypatch,
) -> None:
    """Regression witness: the pre-fix geometry quarters whole expert scales.

    ``use_ep=False`` with a whole-expert local partition is exactly the
    geometry resolution the PR16 report hit (dest 2304 != loaded 576); the
    loader must keep raising rather than silently writing a quarter-width row.
    """
    whole_intermediate = HIDDEN * WHOLE_EXPERTS
    owner, method = _moe_setup(
        monkeypatch, use_ep=False, intermediate=whole_intermediate, tp_size=WHOLE_EXPERTS
    )
    quarter = whole_intermediate // WHOLE_EXPERTS

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "EXL3 aligned buffer mismatch experts.w13_svh shard=w1 expert=0: "
            "dest (128,) != src (64,)"
        ),
    ):
        method._load_exl3(
            owner.w13_svh,
            torch.arange(whole_intermediate, dtype=torch.float16),
            "experts.w13_svh",
            shard_id="w1",
            expert_id=0,
            return_success=True,
        )


def test_pure_tp_still_slices_expert_features(monkeypatch):
    """Without EP the checkpoint is global; the TP-wide window lands at the
    head of the aligned buffer (pad zero) and a full-width un-narrowed
    tensor is accepted verbatim by the loader fallback."""
    partition = HIDDEN * 2
    owner, method = _moe_setup(
        monkeypatch, use_ep=False, intermediate=partition, tp_size=WHOLE_EXPERTS
    )
    global_svh = torch.arange(partition * WHOLE_EXPERTS, dtype=torch.float16)

    ok = method._load_exl3(
        owner.w13_svh,
        global_svh,
        "experts.w13_svh",
        shard_id="w1",
        expert_id=0,
        return_success=True,
    )

    assert ok is True
    row = owner.w13_svh.data[0, 0]
    torch.testing.assert_close(row[:partition], global_svh[:partition], check_dtype=False)
    torch.testing.assert_close(row, global_svh, check_dtype=False)


def test_wrong_rank_expert_is_rejected_under_ep(monkeypatch):
    """The ownership filter stays authoritative: no write for other-rank experts."""
    whole_intermediate = HIDDEN * WHOLE_EXPERTS
    owner, method = _moe_setup(
        monkeypatch, use_ep=True, intermediate=whole_intermediate, tp_size=WHOLE_EXPERTS
    )
    before = owner.w13_svh.detach().clone()
    loaded = torch.arange(whole_intermediate, dtype=torch.float16)

    assert (
        method._load_exl3(
            owner.w13_svh,
            loaded,
            "experts.w13_svh",
            shard_id="w1",
            expert_id=1,
            return_success=True,
        )
        is False
    )
    assert (
        method._load_exl3(
            owner.w13_svh,
            loaded,
            "experts.w13_svh",
            shard_id="w1",
            expert_id=1,
        )
        is None
    )
    torch.testing.assert_close(owner.w13_svh.detach(), before, equal_nan=True)
