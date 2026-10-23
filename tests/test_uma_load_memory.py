"""CPU tests for vllm_exl3.uma_load_memory (load-time trim + post-load embed alias)."""

from __future__ import annotations

import types

import pytest

from vllm_exl3 import uma_load_memory as uma


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (
        "VLLM_EXL3_TRIM_AFTER_LAYER",
        "VLLM_EXL3_UMA_POSTLOAD",
        "VLLM_EXL3_MEM_PROBE",
    ):
        monkeypatch.delenv(k, raising=False)
    for k in uma.STATS:
        uma.STATS[k] = 0
    yield


def _fake_exl3(calls):
    def prepare_trellis_arena_plan(layer, shapes_by_proj):
        calls.append(("plan", layer, shapes_by_proj))
        return {"ok": True}

    return types.SimpleNamespace(prepare_trellis_arena_plan=prepare_trellis_arena_plan)


def test_trim_runs_before_each_arena_plan_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(uma, "trim", lambda: calls.append(("trim",)))
    ex = _fake_exl3(calls)
    uma.install_uma_load_memory(ex)
    monkeypatch.setenv("VLLM_EXL3_TRIM_AFTER_LAYER", "1")
    layer = types.SimpleNamespace(layer_name="model.layers.3.mlp.experts")
    assert ex.prepare_trellis_arena_plan(layer, {"gate": {}}) == {"ok": True}
    assert [c[0] for c in calls] == ["trim", "plan"]
    assert uma.STATS["layer_trims"] == 1


def test_trim_is_inert_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(uma, "trim", lambda: calls.append(("trim",)))
    ex = _fake_exl3(calls)
    uma.install_uma_load_memory(ex)
    ex.prepare_trellis_arena_plan(object(), {})
    assert [c[0] for c in calls] == ["plan"]
    assert uma.STATS["layer_trims"] == 0


def test_install_is_idempotent():
    calls = []
    ex = _fake_exl3(calls)
    uma.install_uma_load_memory(ex)
    first = ex.prepare_trellis_arena_plan
    uma.install_uma_load_memory(ex)
    assert ex.prepare_trellis_arena_plan is first


def _loader_cls(model):
    class Loader:
        def load_model(self, *a, **k):
            return model

    return Loader


def test_postload_aliases_embedding_and_trims(monkeypatch):
    from vllm_exl3 import ats_alias

    model = object()
    seen = []
    monkeypatch.setattr(ats_alias, "alias_embedding", lambda m: seen.append(m) or 1)
    monkeypatch.setattr(uma, "trim", lambda: seen.append("trim"))
    Loader = _loader_cls(model)
    uma.install_postload_hook(Loader)
    uma.install_postload_hook(Loader)  # idempotent
    monkeypatch.setenv("VLLM_EXL3_UMA_POSTLOAD", "1")
    assert Loader().load_model() is model
    assert seen == [model, "trim"]
    assert uma.STATS["embed_aliased"] == 1 and uma.STATS["postload_runs"] == 1


def test_postload_inert_by_default(monkeypatch):
    from vllm_exl3 import ats_alias

    seen = []
    monkeypatch.setattr(ats_alias, "alias_embedding", lambda m: seen.append(m) or 1)
    Loader = _loader_cls(object())
    uma.install_postload_hook(Loader)
    Loader().load_model()
    assert seen == [] and uma.STATS["postload_runs"] == 0


def test_postload_alias_failure_never_fails_the_load(monkeypatch):
    from vllm_exl3 import ats_alias

    def boom(_m):
        raise RuntimeError("offset not aligned")

    monkeypatch.setattr(ats_alias, "alias_embedding", boom)
    monkeypatch.setattr(uma, "trim", lambda: None)
    model = object()
    Loader = _loader_cls(model)
    uma.install_postload_hook(Loader)
    monkeypatch.setenv("VLLM_EXL3_UMA_POSTLOAD", "1")
    assert Loader().load_model() is model
    assert uma.STATS["embed_aliased"] == 0


def test_mem_line_shape():
    line = uma.mem_line("X")
    assert line.startswith("MEMPROBE X cuda_alloc=") and "avail=" in line and "rss_anon=" in line


def test_probe_defaults_follow_switches(monkeypatch):
    assert not uma.probe_enabled()
    monkeypatch.setenv("VLLM_EXL3_TRIM_AFTER_LAYER", "1")
    assert uma.probe_enabled()
    monkeypatch.setenv("VLLM_EXL3_MEM_PROBE", "0")
    assert not uma.probe_enabled()


def test_register_installs_the_trim_wrapper():
    import vllm_exl3
    from vllm_exl3 import exl3

    vllm_exl3.register()
    assert getattr(exl3.prepare_trellis_arena_plan, "_uma_trim", False)
