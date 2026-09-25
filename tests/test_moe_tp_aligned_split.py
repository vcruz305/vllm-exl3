from __future__ import annotations

import pytest
import torch

from vllm_exl3.exl3 import _narrow_tp, aligned_tp_split, shard_exl3_col, shard_exl3_row


def test_split_dsv41_intermediate_tp4_is_block_aligned() -> None:
    shards = [aligned_tp_split(2304, r, 4, 128) for r in range(4)]
    assert shards == [(0, 640), (640, 640), (1280, 512), (1792, 512)]
    for off, length in shards:
        assert off % 128 == 0 and length % 128 == 0


def test_split_matches_equal_chunks_when_divisible() -> None:
    assert [aligned_tp_split(2304, r, 2, 128) for r in range(2)] == [(0, 1152), (1152, 1152)]


def test_split_rejects_unaligned_size() -> None:
    with pytest.raises(ValueError, match="not a multiple of 128"):
        aligned_tp_split(2300, 0, 4, 128)


def test_narrow_tp_is_unchanged_unless_aligned_requested(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_EXL3_MOE_TP_ALIGN", "128")
    t = torch.arange(2304)
    assert _narrow_tp(t, 0, 0, 4).numel() == 576
    assert _narrow_tp(t, 0, 0, 4, aligned=True).numel() == 640


def test_narrow_tp_env_off_keeps_equal_split(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_EXL3_MOE_TP_ALIGN", raising=False)
    t = torch.arange(2304)
    assert _narrow_tp(t, 0, 3, 4, aligned=True).tolist() == list(range(1728, 2304))


def test_trellis_tiles_and_scales_shard_consistently(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_EXL3_MOE_TP_ALIGN", "128")
    hidden_tiles, inter, k_words = 4, 2304, 48
    # gate/up: trellis [in_tiles, out_tiles, K], svh [inter]
    trellis = torch.arange(hidden_tiles * (inter // 16) * k_words).view(hidden_tiles, inter // 16, k_words)
    svh = torch.arange(inter)
    for rank, (off, length) in enumerate(aligned_tp_split(inter, r, 4, 128) for r in range(4)):
        t = shard_exl3_col(trellis, "trellis", rank, 4, aligned=True)
        s = shard_exl3_col(svh, "svh", rank, 4, aligned=True)
        assert t.shape[1] * 16 == length == s.numel()
        assert torch.equal(t, trellis[:, off // 16 : (off + length) // 16])
        assert s[0].item() == off
    # down: trellis [inter_tiles, out_tiles, K], suh [inter]
    down = torch.arange((inter // 16) * hidden_tiles * k_words).view(inter // 16, hidden_tiles, k_words)
    off, length = aligned_tp_split(inter, 2, 4, 128)
    d = shard_exl3_row(down, "trellis", 2, 4, aligned=True)
    assert torch.equal(d, down[off // 16 : (off + length) // 16])
    assert shard_exl3_row(svh, "suh", 2, 4, aligned=True).numel() == length


def test_unaligned_geometry_is_refused_not_silently_split(monkeypatch) -> None:
    """An unalignable size must raise, not fall back to the equal split.

    The equal split is what cuts a Hadamard block and decodes every shard
    against the wrong transform. Falling back to it silently is the failure
    this PR exists to remove, so with the env var set the load has to stop.
    """
    monkeypatch.setenv("VLLM_EXL3_MOE_TP_ALIGN", "128")
    t = torch.arange(2300)
    with pytest.raises(RuntimeError, match="not a multiple"):
        _narrow_tp(t, 0, 0, 4, 1, aligned=True)


def test_aligned_geometry_does_not_raise(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_EXL3_MOE_TP_ALIGN", "128")
    t = torch.arange(2304)
    assert _narrow_tp(t, 0, 2, 4, 1, aligned=True).numel() == 512


def test_rotation_balances_chunk_sizes_across_layers(monkeypatch) -> None:
    from types import SimpleNamespace

    from vllm_exl3.exl3 import moe_tp_rotation

    monkeypatch.setenv("VLLM_EXL3_MOE_TP_ROTATE", "1")
    totals = [0] * 4
    for layer_idx in range(40):
        layer = SimpleNamespace(layer_name=f"model.layers.{layer_idx}.ffn.experts")
        rot = moe_tp_rotation(layer, 4)
        for rank in range(4):
            totals[rank] += aligned_tp_split(2304, (rank + rot) % 4, 4, 128)[1]
    assert totals == [576 * 40] * 4


def test_rotation_off_by_default(monkeypatch) -> None:
    from types import SimpleNamespace

    from vllm_exl3.exl3 import moe_tp_rotation

    monkeypatch.delenv("VLLM_EXL3_MOE_TP_ROTATE", raising=False)
    assert moe_tp_rotation(SimpleNamespace(layer_name="model.layers.7.ffn.experts"), 4) == 0
