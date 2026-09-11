from types import SimpleNamespace

import pytest

from vllm_exl3.deepseek_v41 import (
    V41_NATIVE_MOE_ABI,
    install_deepseek_v41_compat,
    is_deepseek_v41_source_quant,
    native_p2b_geometry_supported,
    plan_deepseek_v41,
    should_delegate_dspark_source,
    source_weight_block_size,
)


def _v41_config(**overrides):
    values = {
        "non_routed_quantization": {
            "quant_method": "deepseek_v4_fp8",
            "weight_block_size": [32, 32],
            "activation_scheme": "dynamic",
        },
        "mtp_experts": "source",
        "mtp_experts_start_layer": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_tp4_ep4_is_the_fused_first_boot_layout():
    plan = plan_deepseek_v41()
    assert plan.tensor_parallel_size == 4
    assert plan.expert_parallel is True
    assert plan.moe_tensor_parallel_size == 1
    assert plan.expert_parallel_size == 4
    assert plan.global_experts == 384
    assert plan.local_experts == 96
    assert plan.hidden_size == 5120
    assert plan.intermediate_size_per_rank == 2304
    assert plan.top_k == 6
    assert plan.exllamav3_fused_candidate is True
    assert plan.native_p2b_candidate is True
    assert plan.native_p2b_requires_abi == V41_NATIVE_MOE_ABI == 3
    assert plan.native_p2b_default_enabled is False
    assert plan.preferred_first_boot_backend == "exllamav3"


def test_pure_tp4_exposes_the_576_wide_problem():
    plan = plan_deepseek_v41(expert_parallel=False)
    assert plan.moe_tensor_parallel_size == 4
    assert plan.expert_parallel_size == 1
    assert plan.local_experts == 384
    assert plan.intermediate_size_per_rank == 576
    assert plan.exllamav3_fused_candidate is False
    assert plan.native_p2b_candidate is False
    assert plan.preferred_first_boot_backend == "loop"


def test_native_geometry_requires_128_aligned_dimensions():
    assert native_p2b_geometry_supported(5120, 2304) is True
    assert native_p2b_geometry_supported(4096, 2048) is True
    assert native_p2b_geometry_supported(5120, 576) is False
    assert native_p2b_geometry_supported(0, 2304) is False


def test_source_quantization_traits_surface_v41_mxfp8_block_shape():
    config = _v41_config()
    assert source_weight_block_size(config) == [32, 32]
    assert is_deepseek_v41_source_quant(config) is True


def test_source_quantization_rejects_non_v41_delegate_shape():
    config = _v41_config(
        non_routed_quantization={
            "quant_method": "deepseek_v4_fp8",
            "weight_block_size": [128, 128],
        }
    )
    assert source_weight_block_size(config) == [128, 128]
    assert is_deepseek_v41_source_quant(config) is False


def test_dspark_source_delegation_can_be_inferred_from_128_expert_stage():
    config = _v41_config()
    assert should_delegate_dspark_source(
        config,
        prefix="model.layers.40.ffn.experts",
        global_num_experts=128,
    ) is True
    assert should_delegate_dspark_source(
        config,
        prefix="model.layers.39.ffn.experts",
        global_num_experts=384,
    ) is False


def test_explicit_dspark_start_layer_remains_authoritative():
    config = _v41_config(mtp_experts_start_layer=40)
    assert should_delegate_dspark_source(
        config,
        prefix="model.layers.39.ffn.experts",
        global_num_experts=128,
    ) is False
    assert should_delegate_dspark_source(
        config,
        prefix="model.layers.40.ffn.experts",
        global_num_experts=128,
    ) is True


def test_dspark_inference_fails_closed_without_source_request():
    config = _v41_config(mtp_experts="exl3")
    assert should_delegate_dspark_source(
        config,
        prefix="model.layers.40.ffn.experts",
        global_num_experts=128,
    ) is False


def test_installer_surfaces_outer_weight_block_size_and_is_idempotent():
    class FakeConfig:
        def __init__(self):
            self.non_routed_quantization = {
                "quant_method": "deepseek_v4_fp8",
                "weight_block_size": [32, 32],
            }

        def get_quant_method(self, layer, prefix):
            return None

    def existing_dimensions_supported(x2d, layer, inners, limit=None):
        return False

    module = SimpleNamespace(
        Exl3Config=FakeConfig,
        _native_moe_dimensions_supported=existing_dimensions_supported,
    )
    install_deepseek_v41_compat(module)
    wrapped_dimensions = module._native_moe_dimensions_supported
    install_deepseek_v41_compat(module)

    config = FakeConfig()
    assert config.weight_block_size == [32, 32]
    assert config.has_blocked_weights() is True
    assert module._vllm_exl3_v41_compat_installed is True
    assert wrapped_dimensions is module._native_moe_dimensions_supported


def test_plan_rejects_non_divisible_expert_layout():
    with pytest.raises(ValueError, match="global_experts"):
        plan_deepseek_v41(global_experts=385)
