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
        "VLLM_EXL3_DRAFT_VOCAB_META",
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


def _loader_cls(model, on_load=None):
    class Loader:
        def load_model(self, vllm_config=None, model_config=None, prefix=""):
            if on_load is not None:
                on_load()
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


def test_tensor_census_counts_cpu_storages_once():
    torch = pytest.importorskip("torch")
    a = torch.zeros(1024, 1024, dtype=torch.int16)
    b = a[:10]  # same storage
    line = uma.tensor_census()
    assert line.startswith("TENSORS cpu_storages=") and "(1024, 1024):int16x1=2MiB" in line
    del a, b


def test_diagnostic_abort_after_n_layers(monkeypatch):
    calls = []
    monkeypatch.setattr(uma, "trim", lambda: None)
    ex = _fake_exl3(calls)
    uma.install_uma_load_memory(ex)
    monkeypatch.setenv("VLLM_EXL3_TRIM_AFTER_LAYER", "1")
    monkeypatch.setenv("VLLM_EXL3_MEM_PROBE", "0")
    monkeypatch.setenv("VLLM_EXL3_MEM_PROBE_ABORT_AFTER", "2")
    ex.prepare_trellis_arena_plan(object(), {})
    with pytest.raises(RuntimeError, match="diagnostic load stop"):
        ex.prepare_trellis_arena_plan(object(), {})


def _write_safetensors(path, names):
    import json
    import struct

    header = {n: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]} for n in names}
    raw = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        fh.write(b"\0\0")


def _draft_cfg(model_dir, arch="DSparkMiniMaxDraftModel"):
    return types.SimpleNamespace(
        model=str(model_dir), hf_config=types.SimpleNamespace(architectures=[arch])
    )


def test_checkpoint_vocab_scan(tmp_path):
    _write_safetensors(tmp_path / "a.safetensors", ["model.layers.0.mlp.w1.weight"])
    assert uma.checkpoint_has_vocab_tables(str(tmp_path)) is False
    _write_safetensors(tmp_path / "b.safetensors", ["lm_head.weight"])
    assert uma.checkpoint_has_vocab_tables(str(tmp_path)) is True
    assert uma.checkpoint_has_vocab_tables(str(tmp_path / "missing")) is None


def test_is_vocab_shared_draft_requires_arch_and_absent_tables(tmp_path):
    _write_safetensors(tmp_path / "d.safetensors", ["language_model.model.dspark.fc.weight"])
    assert uma.is_vocab_shared_draft(_draft_cfg(tmp_path)) is True
    assert uma.is_vocab_shared_draft(_draft_cfg(tmp_path, "MiniMaxM3SparseForConditionalGeneration")) is False
    _write_safetensors(tmp_path / "e.safetensors", ["model.embed_tokens.weight"])
    assert uma.is_vocab_shared_draft(_draft_cfg(tmp_path)) is False


class _FakeEmbedMethod:
    def create_weights(self, layer, rows, cols):
        import torch

        layer.weight = torch.empty(rows, cols, dtype=torch.bfloat16)


def test_draft_vocab_tables_built_on_meta_only_inside_draft_load(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    _write_safetensors(tmp_path / "d.safetensors", ["language_model.model.dspark.fc.weight"])
    built = {}

    def on_load():
        layer = types.SimpleNamespace()
        _FakeEmbedMethod().create_weights(layer, 200064, 64)
        built["layer"] = layer

    Loader = _loader_cls(object(), on_load)
    uma.install_postload_hook(Loader, embedding_method_cls=_FakeEmbedMethod)
    monkeypatch.setenv("VLLM_EXL3_DRAFT_VOCAB_META", "1")
    Loader().load_model(None, _draft_cfg(tmp_path))
    assert built["layer"].weight.device.type == "meta"
    assert built["layer"]._exl3_meta_vocab_placeholder is True
    assert uma.STATS["draft_meta_vocab"] == 1
    # restored afterwards: a later (target) build allocates for real
    later = types.SimpleNamespace()
    _FakeEmbedMethod().create_weights(later, 4, 4)
    assert later.weight.device.type == "cpu"


def test_draft_with_its_own_tables_allocates_normally(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    _write_safetensors(tmp_path / "d.safetensors", ["lm_head.weight"])
    built = {}

    def on_load():
        layer = types.SimpleNamespace()
        _FakeEmbedMethod().create_weights(layer, 8, 8)
        built["layer"] = layer

    Loader = _loader_cls(object(), on_load)
    uma.install_postload_hook(Loader, embedding_method_cls=_FakeEmbedMethod)
    monkeypatch.setenv("VLLM_EXL3_DRAFT_VOCAB_META", "1")
    Loader().load_model(None, _draft_cfg(tmp_path))
    assert built["layer"].weight.device.type == "cpu"


def test_draft_load_never_runs_the_embedding_alias(monkeypatch, tmp_path):
    from vllm_exl3 import ats_alias

    seen = []
    monkeypatch.setattr(ats_alias, "alias_embedding", lambda m: seen.append(m) or 1)
    monkeypatch.setattr(uma, "trim", lambda: seen.append("trim"))
    _write_safetensors(tmp_path / "d.safetensors", ["x.weight"])
    Loader = _loader_cls(object())
    uma.install_postload_hook(Loader, embedding_method_cls=_FakeEmbedMethod)
    monkeypatch.setenv("VLLM_EXL3_UMA_POSTLOAD", "1")
    Loader().load_model(None, _draft_cfg(tmp_path))
    assert seen == ["trim"]
