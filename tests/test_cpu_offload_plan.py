from vllm_exl3.cpu_offload import plan_exllamav3_cpu_offload


def test_v41_missing_architecture_blocks_cpu_offload():
    plan = plan_exllamav3_cpu_offload(
        architecture="DeepseekV41ForCausalLM",
        codebooks="mul1",
        max_k=4,
        uniform_expert_biases=True,
        exllamav3_architecture_available=False,
    )
    assert not plan.exllamav3_metadata_candidate
    assert plan.vllm_exl3_execution_available is False
    assert any("cannot instantiate" in item for item in plan.blockers)


def test_mcg_is_not_upstream_cpu_moe_candidate():
    plan = plan_exllamav3_cpu_offload(
        architecture="DeepseekV41ForCausalLM",
        codebooks="mcg",
        max_k=4,
        uniform_expert_biases=True,
        exllamav3_architecture_available=True,
    )
    assert not plan.exllamav3_metadata_candidate
    assert any("mul1-only" in item for item in plan.blockers)


def test_mul1_k8_can_reach_conditional_metadata_state():
    plan = plan_exllamav3_cpu_offload(
        architecture="DeepseekV41ForCausalLM",
        codebooks=["mul1"],
        max_k=8,
        uniform_expert_biases=True,
        exllamav3_architecture_available=True,
    )
    assert plan.exllamav3_metadata_candidate
    assert plan.blockers == ()
    assert plan.vllm_exl3_execution_available is False


def test_unknown_bias_stays_warning_not_false_pass_claim():
    plan = plan_exllamav3_cpu_offload(
        architecture="DeepseekV41ForCausalLM",
        codebooks="mul1",
        max_k=4,
        uniform_expert_biases=None,
        exllamav3_architecture_available=True,
    )
    assert plan.exllamav3_metadata_candidate
    assert any("bias" in item for item in plan.warnings)
    assert plan.vllm_exl3_execution_available is False


def test_k_above_limit_blocks():
    plan = plan_exllamav3_cpu_offload(
        architecture="DeepseekV41ForCausalLM",
        codebooks="mul1",
        max_k=9,
        uniform_expert_biases=True,
        exllamav3_architecture_available=True,
    )
    assert not plan.exllamav3_metadata_candidate
    assert any("exceeds" in item for item in plan.blockers)
