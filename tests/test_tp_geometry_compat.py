from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_exl3.tp_geometry_compat import (
    install_tp_geometry_compat,
    nested_moe_tp_geometry,
)


def _owner(tp_rank: int, tp_size: int):
    parallel = SimpleNamespace(tp_rank=tp_rank, tp_size=tp_size)
    config = SimpleNamespace(moe_parallel_config=parallel)
    return SimpleNamespace(moe_config=config)


def test_nested_geometry_prefers_ep_local_tp1() -> None:
    assert nested_moe_tp_geometry(_owner(0, 1)) == (0, 1)


def test_nested_geometry_preserves_tp_rank() -> None:
    assert nested_moe_tp_geometry(_owner(2, 4)) == (2, 4)


def test_nested_geometry_accepts_fused_moe_config_directly() -> None:
    parallel = SimpleNamespace(tp_rank=1, tp_size=2)
    config = SimpleNamespace(moe_parallel_config=parallel)
    assert nested_moe_tp_geometry(config) == (1, 2)


def test_nested_geometry_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError, match="invalid vLLM MoE TP geometry"):
        nested_moe_tp_geometry(_owner(2, 2))


def test_installed_wrapper_falls_back_to_existing_resolver() -> None:
    calls = []

    def original(*owners):
        calls.append(owners)
        return (7, 8)

    fake = SimpleNamespace(_resolve_tp_geometry=original)
    install_tp_geometry_compat(fake)

    # Nested current-vLLM geometry takes precedence and never consults process TP.
    assert fake._resolve_tp_geometry(_owner(0, 1)) == (0, 1)
    assert calls == []

    # Older/other owners retain the existing resolver behavior.
    plain = SimpleNamespace()
    assert fake._resolve_tp_geometry(plain) == (7, 8)
    assert calls == [(plain,)]

    # Installation is idempotent.
    wrapped = fake._resolve_tp_geometry
    install_tp_geometry_compat(fake)
    assert fake._resolve_tp_geometry is wrapped
