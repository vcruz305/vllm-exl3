"""CPU regression tests for EXL3 expert-map and TP loader geometry."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import vllm_exl3.exl3 as exl3


class _ReadOnlyExpertMapLayer:
    def __init__(self, expert_map: torch.Tensor) -> None:
        self._raw_expert_map = expert_map

    @property
    def expert_map(self) -> torch.Tensor:
        return self._raw_expert_map


class _MoEOwner:
    tp_rank = 0
    tp_size = 1

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        return expert_id


def test_moe_loader_prefers_layer_tp_geometry() -> None:
    """A MoE layer with TP=1 must not inherit process TP=8 slicing."""
    owner = _MoEOwner()
    param = torch.nn.Parameter(
        torch.empty(1, 2, 1, 1, 32, dtype=torch.int16), requires_grad=False
    )
    param._exl3_owner = owner
    loaded = torch.arange(32, dtype=torch.int16).reshape(1, 1, 32)
    method = object.__new__(exl3.Exl3MoEMethod)
    method._load_exl3(
        param,
        loaded,
        "experts.w13_trellis",
        shard_id="w1",
        expert_id=0,
    )

    torch.testing.assert_close(param[0, 0], loaded)


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


def _mismatch_param(dest_shape: tuple[int, ...]) -> torch.nn.Parameter:
    """w1 slot-0 parameter whose destination slice has ``dest_shape``."""
    param = torch.nn.Parameter(
        torch.full((1, 2, *dest_shape), 7, dtype=torch.float32),
        requires_grad=False,
    )
    param._exl3_owner = _MoEOwner()
    return param


def test_shape_mismatch_raises_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the opt-in env var a mismatched load must hard-fail untouched."""
    monkeypatch.setenv("VLLM_EXL3_ALLOW_SHAPE_MISMATCH", "0")
    param = _mismatch_param((3, 4))
    method = object.__new__(exl3.Exl3MoEMethod)

    with pytest.raises(RuntimeError, match="shape mismatch"):
        method._load_exl3(
            param,
            torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "experts.w13_trellis",
            shard_id="w1",
            expert_id=0,
            return_success=True,
        )

    assert (param.data[0, 0] == 7).all()


def test_diagnostic_mode_source_smaller_zero_fills_and_coordinate_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in padding: overlap is coordinate-copied, the tail zero-filled."""
    monkeypatch.setenv("VLLM_EXL3_ALLOW_SHAPE_MISMATCH", "1")
    param = _mismatch_param((3, 4))
    src = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 1
    method = object.__new__(exl3.Exl3MoEMethod)

    result = method._load_exl3(
        param, src, "experts.w13_trellis",
        shard_id="w1", expert_id=0, return_success=True,
    )

    assert result is True
    torch.testing.assert_close(param.data[0, 0, :2], src)
    assert (param.data[0, 0, 2:] == 0).all()


def test_diagnostic_mode_source_larger_coordinate_trims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in trimming: only the leading per-dimension overlap is kept."""
    monkeypatch.setenv("VLLM_EXL3_ALLOW_SHAPE_MISMATCH", "1")
    param = _mismatch_param((2, 4))
    src = torch.arange(15, dtype=torch.float32).reshape(3, 5) + 1
    method = object.__new__(exl3.Exl3MoEMethod)

    result = method._load_exl3(
        param, src, "experts.w13_trellis",
        shard_id="w1", expert_id=0, return_success=True,
    )

    assert result is True
    torch.testing.assert_close(param.data[0, 0], src[:2, :4])


def test_diagnostic_mode_equal_numel_keeps_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal-numel/different-shape: rows must not be packed across rows.

    A flattened prefix copy would fill ``dest[2]`` from ``src[1]``'s tail;
    the coordinate copy leaves the non-overlapping row zero. The source is
    deliberately non-contiguous with 12 elements, like the destination.
    """
    monkeypatch.setenv("VLLM_EXL3_ALLOW_SHAPE_MISMATCH", "1")
    param = _mismatch_param((3, 4))
    src = (torch.arange(24, dtype=torch.float32).reshape(4, 6) + 1)[::2]
    assert not src.is_contiguous()
    assert tuple(src.shape) == (2, 6)
    method = object.__new__(exl3.Exl3MoEMethod)

    method._load_exl3(
        param, src, "experts.w13_trellis", shard_id="w1", expert_id=0,
    )

    torch.testing.assert_close(param.data[0, 0, :2], src[:, :4])
    assert (param.data[0, 0, 2:] == 0).all()


def test_diagnostic_mode_dtype_converted_on_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in copy narrows the source dtype to the destination dtype."""
    monkeypatch.setenv("VLLM_EXL3_ALLOW_SHAPE_MISMATCH", "1")
    param = torch.nn.Parameter(
        torch.full((1, 2, 3, 4), 7, dtype=torch.float16), requires_grad=False
    )
    param._exl3_owner = _MoEOwner()
    src = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 1
    method = object.__new__(exl3.Exl3MoEMethod)

    method._load_exl3(
        param, src, "experts.w13_trellis", shard_id="w1", expert_id=0,
    )

    torch.testing.assert_close(
        param.data[0, 0, :2], src.to(torch.float16), rtol=0, atol=0
    )
    assert (param.data[0, 0, 2:] == 0).all()


def test_diagnostic_mode_ndim_mismatch_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank mismatches stay a hard error even in diagnostic mode."""
    monkeypatch.setenv("VLLM_EXL3_ALLOW_SHAPE_MISMATCH", "1")
    param = _mismatch_param((3, 4))
    method = object.__new__(exl3.Exl3MoEMethod)

    with pytest.raises(RuntimeError, match="matching rank"):
        method._load_exl3(
            param,
            torch.arange(12, dtype=torch.float32).reshape(3, 4, 1),
            "experts.w13_trellis",
            shard_id="w1",
            expert_id=0,
            return_success=True,
        )

    assert (param.data[0, 0] == 7).all()


def test_return_success_only_in_diagnostic_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mismatched loads report success only with the explicit opt-in."""
    param = _mismatch_param((3, 4))
    mismatched = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    method = object.__new__(exl3.Exl3MoEMethod)

    monkeypatch.setenv("VLLM_EXL3_ALLOW_SHAPE_MISMATCH", "1")
    assert method._load_exl3(
        param, mismatched, "experts.w13_trellis",
        shard_id="w1", expert_id=0, return_success=True,
    )
    monkeypatch.setenv("VLLM_EXL3_ALLOW_SHAPE_MISMATCH", "0")
    with pytest.raises(RuntimeError, match="shape mismatch"):
        method._load_exl3(
            param, mismatched, "experts.w13_trellis",
            shard_id="w1", expert_id=0, return_success=True,
        )

    # Matching shapes succeed regardless of the diagnostic opt-in.
    matched_param = torch.nn.Parameter(
        torch.zeros(1, 2, 2, 4, dtype=torch.float32), requires_grad=False
    )
    matched_param._exl3_owner = _MoEOwner()
    matched = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    assert method._load_exl3(
        matched_param, matched, "experts.w13_trellis",
        shard_id="w1", expert_id=0, return_success=True,
    )
    assert (
        method._load_exl3(
            matched_param, matched, "experts.w13_trellis",
            shard_id="w1", expert_id=0,
        )
        is None
    )
