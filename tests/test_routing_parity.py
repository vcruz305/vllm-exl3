"""CPU regression tests for native fused-MoE routing shape and accumulation."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import vllm_exl3.exl3 as exl3


class _RecordingNativeExtension:
    """Small CPU stand-in for the native ABI used by the routing wrapper."""

    def __init__(self, abi_version: int = 1) -> None:
        self.P2B_MOE_ABI_VERSION = abi_version
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.options: list[tuple] = []

    def p2b_fused_moe(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        *args,
    ) -> torch.Tensor:
        # The routing arguments are the final four ABI values before the
        # activation flags: expert IDs, routing weights, and three K values.
        ids, weights = args[9], args[10]
        self.options.append(args[11:])
        self.calls.append((ids.detach().clone(), weights.detach().clone()))
        assert ids.ndim == 1
        assert weights.ndim == 1
        assert ids.shape == weights.shape
        contribution = (ids.to(torch.float32) + 1.0) * weights.to(torch.float32)
        out.copy_(x.to(torch.float32) * contribution.sum())
        return out


@pytest.mark.parametrize("tokens", [1, 2, 4, 8])
@pytest.mark.parametrize("top_k", [1, 2, 3, 6, 8])
@pytest.mark.parametrize(
    "intermediate,limit,bits,abi_version",
    [(2048, None, 4, 1), (2048, None, 2, 2), (2048, 10.0, 3, 2),
     (1024, None, 4, 2), (1024, 10.0, 4, 2)],
)
def test_native_routing_preserves_topk_and_all_contributions(
    monkeypatch: pytest.MonkeyPatch,
    tokens: int,
    top_k: int,
    intermediate: int,
    limit: float | None,
    bits: int,
    abi_version: int,
) -> None:
    """Every decode row receives all K routes and their weighted sum."""
    extension = _RecordingNativeExtension(abi_version)
    monkeypatch.setattr(exl3, "_load_native_exl3_ext", lambda: extension)
    monkeypatch.setattr(exl3, "_native_moe_dimensions_supported", lambda *args: True)

    hidden = 3
    n_experts = max(top_k + 2, 16)
    x = torch.arange(1, tokens * hidden + 1, dtype=torch.float32).reshape(
        tokens, hidden
    )
    ids = torch.arange(tokens * top_k, dtype=torch.long).reshape(tokens, top_k)
    ids = ids.remainder(n_experts)
    weights = torch.linspace(
        0.07, 0.97, steps=tokens * top_k, dtype=torch.float32
    ).reshape(tokens, top_k)
    layer = SimpleNamespace(
        _exl3_ptrs={
            key: object()
            for key in (
                "gate_trellis",
                "gate_suh",
                "gate_svh",
                "up_trellis",
                "up_suh",
                "up_svh",
                "down_trellis",
                "down_suh",
                "down_svh",
            )
        },
        _exl3_k=bits,
        _exl3_intermediate_local=intermediate,
    )
    inners = [{} for _ in range(n_experts)]

    output = exl3._apply_native_fused_moe(x, ids, weights, layer, inners, None, limit)

    assert output is not None
    assert len(extension.calls) == tokens
    expected_options = (bits, bits, bits, True)
    if abi_version >= 2:
        expected_options += (intermediate, limit or 0.0)
    assert extension.options == [expected_options] * tokens
    for row, (recorded_ids, recorded_weights) in enumerate(extension.calls):
        assert recorded_ids.shape == (top_k,)
        assert recorded_weights.shape == (top_k,)
        torch.testing.assert_close(recorded_ids, ids[row].to(torch.int32))
        torch.testing.assert_close(
            recorded_weights, weights[row].to(torch.float16)
        )

    expected_scale = (
        (ids.to(torch.float32) + 1.0) * weights.to(torch.float16).to(torch.float32)
    ).sum(dim=1)
    expected = x.to(torch.float16).to(torch.float32) * expected_scale.unsqueeze(1)
    expected = expected.to(torch.float16).to(torch.float32)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
