import pytest

from vllm_exl3.prefill_policy import (
    grouped_prefill_max_rows,
    grouped_prefill_scratch_bytes,
    plan_grouped_prefill,
)


def _plan(**overrides):
    values = dict(
        bits=2,
        codebook="mcg",
        has_mul1=False,
        hidden_size=4096,
        intermediate_size=2048,
        routed_rows=4096,
        fat_threshold=256,
        tile_rows=64,
        max_rows=1024,
        requested=True,
    )
    values.update(overrides)
    return plan_grouped_prefill(**values)


@pytest.mark.parametrize("bits", [2, 3])
def test_tp1_k2_k3_candidates_are_eligible(bits):
    plan = _plan(bits=bits)
    assert plan.eligible
    assert plan.reason == "eligible"
    assert plan.window_rows == 1024
    assert plan.windows == 4
    assert plan.scratch_bytes == 1024 * (4096 + 3 * 2048) * 2


def test_policy_is_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_GROUPED_PREFILL", raising=False)
    plan = _plan(requested=None)
    assert not plan.requested
    assert not plan.eligible
    assert plan.reason == "disabled"


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"bits": 4}, "unsupported_bits:4"),
        ({"codebook": "mul1"}, "unsupported_codebook:mul1"),
        ({"has_mul1": True}, "mul1_not_supported"),
        ({"hidden_size": 3968}, "hidden_not_multiple_of_256"),
        ({"intermediate_size": 2000}, "intermediate_not_multiple_of_128"),
        ({"routed_rows": 256}, "below_fat_threshold"),
    ],
)
def test_candidate_contract_fails_closed(overrides, reason):
    plan = _plan(**overrides)
    assert not plan.eligible
    assert plan.reason == reason


def test_large_prefill_is_split_into_bounded_windows():
    plan = _plan(routed_rows=57_344, max_rows=8192)
    assert plan.window_rows == 8192
    assert plan.windows == 7
    assert plan.scratch_bytes == grouped_prefill_scratch_bytes(8192, 4096, 2048)


def test_small_candidate_rounds_workspace_to_tile_without_exceeding_limit():
    plan = _plan(routed_rows=300, max_rows=1024)
    assert plan.window_rows == 320
    assert plan.windows == 1


def test_env_max_rows_is_validated(monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_GROUPED_PREFILL_MAX_ROWS", "4096")
    assert grouped_prefill_max_rows() == 4096
    monkeypatch.setenv("VLLM_EXL3_GROUPED_PREFILL_MAX_ROWS", "bad")
    assert grouped_prefill_max_rows(2048) == 2048
    monkeypatch.setenv("VLLM_EXL3_GROUPED_PREFILL_MAX_ROWS", "0")
    assert grouped_prefill_max_rows(2048) == 2048


@pytest.mark.parametrize(
    "args,exc",
    [
        ((1.5, 4096, 2048), TypeError),
        ((1, -1, 2048), ValueError),
        ((1, 4096, 0), ValueError),
    ],
)
def test_scratch_estimator_rejects_invalid_domains(args, exc):
    with pytest.raises(exc):
        grouped_prefill_scratch_bytes(*args)
