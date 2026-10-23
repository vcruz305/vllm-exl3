"""RED tests for MiniMax-M3 (MiniMax-M3.1-Flash EXL3 SAGE pack) support.

These tests define the milestone-2 contract for serving the
``MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw`` pack with the vllm-exl3 plugin.
They are CPU-only and require neither vLLM nor a GPU: every layer is a
stand-in namespace and every number comes from the parent's pack header
census (57 MoE layers x 128 experts; per-expert K triples (2,2,2) x 6096,
(2,3,2) x 1080, (3,3,3) x 120).

They are RED on purpose: the MiniMax surface below does not exist yet, so
each test fails on the missing behaviour (AssertionError/AttributeError at
test time -- never an import or collection error).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import vllm_exl3.exl3 as exl3
from vllm_exl3 import ats_alias


# ---------------------------------------------------------------------------
# Pack facts (parent-verified header census; CAMPAIGN_STATE.md)
# ---------------------------------------------------------------------------

# Per-expert (w1, w2, w3) K-bit triples of the pack's 7296 routed experts.
PACK_K_CENSUS = {
    (2, 2, 2): 6096,
    (2, 3, 2): 1080,
    (3, 3, 3): 120,
}

# Model-side activation plumbing (vLLM branch transformers_utils/configs/
# minimax_m3.py: alpha 1.702, beta defaults to 1.0, limit 7.0).
MINIMAX_ALPHA = 1.702
MINIMAX_LIMIT = 7.0


def _moe_layer(activation: str, **layer_attrs):
    """RoutedExperts stand-in: vLLM keeps activation on moe_config and the
    swiglu parameters on the layer itself (routed_experts.py stores
    swiglu_limit/alpha/beta as layer attributes)."""
    return SimpleNamespace(
        moe_config=SimpleNamespace(activation=SimpleNamespace(value=activation)),
        **layer_attrs,
    )


# ---------------------------------------------------------------------------
# 1. activation selection: swigluoai(_uninterleave) must reach the exllamav3
#    swiglu_oai kernel (activation_fn_idx 3), not the historical SILU index 0.
# ---------------------------------------------------------------------------


def test_swiglu_oai_kernel_act_constant_exists():
    assert exl3.MOE_ACT_SWIGLU_OAI == 3


def test_swigluoai_uninterleave_maps_to_oai_kernel_act():
    layer = _moe_layer("swigluoai_uninterleave")
    assert exl3.resolve_moe_act(layer) == 3


def test_swigluoai_interleaved_maps_to_oai_kernel_act():
    layer = _moe_layer("swigluoai")
    assert exl3.resolve_moe_act(layer) == 3


def test_silu_activation_stays_on_the_silu_index():
    # Regression guard: the DSV4.1 path must not change.
    assert exl3.resolve_moe_act(_moe_layer("silu")) == exl3.MOE_ACT_SILU == 0


def test_relu2_no_mul_stays_on_the_gateless_index():
    assert (
        exl3.resolve_moe_act(_moe_layer("relu2_no_mul"))
        == exl3.MOE_ACT_RELU2_NOGATE
    )


def test_unknown_gated_activation_is_rejected_not_silenced():
    with pytest.raises(ValueError):
        exl3.resolve_moe_act(_moe_layer("gelu_pytorch_tanh"))


# ---------------------------------------------------------------------------
# 2. alpha/limit selection: the MiniMax gate params must surface the model's
#    (alpha, limit) pair; the historical swiglu_limit path (alpha-less silu,
#    default 10.0) must be untouched.
# ---------------------------------------------------------------------------


def test_minimax_swiglu_defaults_are_the_model_constants():
    assert exl3.MINIMAX_SWIGLU_ALPHA_DEFAULT == pytest.approx(1.702)
    assert exl3.MINIMAX_SWIGLU_LIMIT_DEFAULT == pytest.approx(7.0)


def test_minimax_layer_gate_params_surface_alpha_and_limit():
    layer = _moe_layer(
        "swigluoai_uninterleave",
        swiglu_alpha=MINIMAX_ALPHA,
        swiglu_limit=MINIMAX_LIMIT,
    )
    alpha, limit = exl3.resolve_moe_gate_params(layer)
    assert alpha == pytest.approx(1.702)
    assert limit == pytest.approx(7.0)


def test_minimax_gate_params_fall_back_to_model_defaults():
    # The layer attributes may be absent on older plumbing; the MiniMax model
    # constants, not SWIGLU_LIMIT_DEFAULT (10.0), are the fallback.
    layer = _moe_layer("swigluoai_uninterleave")
    alpha, limit = exl3.resolve_moe_gate_params(layer)
    assert alpha == pytest.approx(1.702)
    assert limit == pytest.approx(7.0)


def test_silu_layer_gate_params_stay_alpha_less():
    # DSV4.1: silu + swiglu_limit, no alpha term in the kernel.
    layer = _moe_layer("silu", swiglu_limit=10.0)
    alpha, limit = exl3.resolve_moe_gate_params(layer)
    assert alpha is None
    assert limit == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 3. expert name mapping: the MiniMax checkpoint keeps per-expert w1/w2/w3
#    tensors under block_sparse_moe.experts; the shard ids keep their
#    gate/up/down meaning.
# ---------------------------------------------------------------------------


def test_minimax_expert_family_is_recognized():
    assert "block_sparse_moe.experts" in exl3.EXPERT_CKPT_PREFIX_FAMILIES


def test_expert_shard_ids_still_map_to_gate_up_down():
    assert exl3._proj_from_shard_id("w1") == "gate"
    assert exl3._proj_from_shard_id("w3") == "up"
    assert exl3._proj_from_shard_id("w2") == "down"


def test_indexer_shards_are_declared_separately_from_qkv():
    assert exl3.MINIMAX_INDEXER_SHARDS == ("index_q", "index_k")


# ---------------------------------------------------------------------------
# 4. non-routed coverage: the pack's quantization_config carries no scope
#    ledger at all ({quant_method: exl3, version: 0.0.2}), so the plugin must
#    derive the EXL3-covered families from the MiniMax architecture itself.
# ---------------------------------------------------------------------------

MINIMAX_EXL3_FAMILIES = (
    "self_attn.qkv_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "block_sparse_moe.shared_experts.gate_proj",
    "block_sparse_moe.shared_experts.up_proj",
    "block_sparse_moe.shared_experts.down_proj",
    "lm_head",
)

MINIMAX_NON_EXL3_FAMILIES = (
    # MXFP8 indexer projections: never EXL3 linear storage.
    "self_attn.index_q_proj",
    "self_attn.index_k_proj",
    # F32 router gate and scoring bias: unquantized.
    "block_sparse_moe.gate",
    "block_sparse_moe.e_score_correction_bias",
    # BF16 embedding and norms: unquantized.
    "embed_tokens",
)


def test_minimax_exl3_coverage_families_are_declared():
    families = exl3.MINIMAX_NON_ROUTED_FAMILIES
    for family in MINIMAX_EXL3_FAMILIES:
        assert family in families, family


def test_minimax_indexer_router_and_embed_families_are_excluded():
    families = exl3.MINIMAX_NON_ROUTED_FAMILIES
    for family in MINIMAX_NON_EXL3_FAMILIES:
        assert family not in families, family


def test_minimax_pack_config_derives_non_routed_coverage():
    # The real pack's quantization_config: no non_routed_exl3 block, no
    # tensor_storage ledger. MiniMax detection must come from the model type,
    # not from the quantization dict.
    cfg = exl3.Exl3Config.from_config(
        {"quant_method": "exl3", "version": "0.0.2"},
    )
    assert exl3.minimax_pack_detected(cfg) is True
    assert cfg.non_routed_exl3, "MiniMax coverage must be derived"


# ---------------------------------------------------------------------------
# 5. per-expert K census: the pack's mixed-K triples must be accepted as-is.
# ---------------------------------------------------------------------------


def test_pack_per_expert_k_census_is_accepted():
    assert exl3.expert_k_census_supported(PACK_K_CENSUS) is True


def test_unsupported_k_census_is_rejected():
    bad = dict(PACK_K_CENSUS)
    bad[(2, 2, 9)] = 1  # K9 is outside the EXL3 K family
    assert exl3.expert_k_census_supported(bad) is False


def test_empty_census_is_rejected():
    assert exl3.expert_k_census_supported({}) is False


# ---------------------------------------------------------------------------
# 6. ATS cold-expert aliasing: the MiniMax expert/embed checkpoint names must
#    be aliasable for the 262K-context NVMe streaming plan.
# ---------------------------------------------------------------------------


def test_ats_alias_recognizes_minimax_expert_prefix():
    assert "block_sparse_moe.experts" in ats_alias.EXPERT_CKPT_PREFIXES


def test_ats_alias_declares_the_minimax_embed_key():
    assert (
        ats_alias.MINIMAX_EMBED_KEY
        == "language_model.model.embed_tokens.weight"
    )
