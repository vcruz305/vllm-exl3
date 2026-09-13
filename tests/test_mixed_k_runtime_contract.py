from __future__ import annotations

import vllm_exl3


def test_mixed_k_runtime_contract_is_explicit() -> None:
    mixed = vllm_exl3.runtime_diagnostics()["mixed_k"]
    assert mixed["tensor_level_mixed_k_within_layer"] is True
    assert mixed["heterogeneous_dispatch"] == "python_loop"
    assert mixed["uniform_k_dispatch"] == "fused_when_available"
    assert mixed["cudagraph_qualified"] is False
    assert mixed["recommended_first_boot"] == "eager"
    assert mixed["arena_prescan_placement_contract"] == (
        "linear_contiguous_global_expert_ids"
    )
    assert mixed["legacy_shape_overlap_diagnostic"] is False
