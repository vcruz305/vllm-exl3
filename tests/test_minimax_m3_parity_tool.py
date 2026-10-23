"""Milestone-3 tests: parity tool self-test, provisional alias selection,
stats-based alias regeneration for the MiniMax expert family, the MiniMax
route-stats recipe, and the DSpark draft-quantization decline contract.

CPU-only, no vLLM, no GPU (the parity tool's GPU arms run in the dev image).
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vllm_exl3.exl3 as exl3

REPO = Path(__file__).resolve().parents[1]

# ── GPU parity tool: the CPU self-test must pass as a subprocess ────────────


def test_parity_tool_cpu_selftest_passes():
    proc = subprocess.run(
        [sys.executable, "tools/gb10_minimax_m3_parity.py", "--cpu-selftest"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "GB10_MINIMAX_M3_PARITY SELFTEST PASS" in proc.stdout
    assert "no hardware parity claim" in proc.stdout


def test_parity_tool_case_b_offsets_are_the_branch_split():
    sys.path.insert(0, str(REPO / "tools"))
    try:
        import gb10_minimax_m3_parity as tool
    finally:
        sys.path.pop(0)
    # vllm linear.py fused split: [q | k | v | index_q | index_k] with
    # output sizes 8192/512/512/512/128 at TP1.
    assert tool.B_OUTS == (8192, 512, 512, 512, 128)
    offs = list(tool.B_OFFSETS) + [sum(tool.B_OUTS)]
    assert [offs[i + 1] - offs[i] for i in range(5)] == list(tool.B_OUTS)
    # Case A covers exactly the pack's per-expert K census.
    assert tool.MEM_BUDGET_GIB == 4.0


# ── Codebook contract: run-1 Case B failed on a codebook mismatch ──────────
#
# The pack's EXL3 tensors carry the mul1 marker (census: a `.mul1` I32 next
# to every trellis/suh/svh triple); the loader validates it and every
# production decode path resolves the mul1 codebook. Run 1 of GPU_JOB-2
# built markerless synthetic packs and an MCG-codebook reference, so the
# layer decoded the plain codebook while the reference decoded MCG --
# cosine ~0 by construction, not a kernel defect.


def _tool():
    sys.path.insert(0, str(REPO / "tools"))
    try:
        import gb10_minimax_m3_parity as tool
    finally:
        sys.path.pop(0)
    return tool


def test_gemv_reference_defaults_to_the_mul1_codebook():
    import inspect

    tool = _tool()
    # The native GEMV entry maps mcg=False to the mul1 decoder; the
    # reference must default to it (run 1 hardcoded True = the MCG
    # codebook, uncorrelated with the layer output).
    sig = inspect.signature(tool._gemv_ref)
    assert "mcg" in sig.parameters
    assert sig.parameters["mcg"].default is False


def test_synthetic_shards_carry_the_loader_markers():
    tool = _tool()
    # The loader's per-shard marker value, and the post-load launch flags
    # for a mul1 pack: (mcg, mul1) = (False, True) per projection.
    marker = tool._mul1_marker(torch.device("cpu"))
    assert int(marker[0].item()) == exl3.MUL1_MARKER_SIGNED_INT32
    assert tool._loader_codebook_flags([(False, True)] * 3) == (False, True) * 3
    with pytest.raises(ValueError):
        tool._loader_codebook_flags([(False, True)] * 2)
    # Case A's fake layer records the loader's flags for marker-carrying
    # inners (the MCG legacy default would test the wrong codebook).
    src = (REPO / "tools/gb10_minimax_m3_parity.py").read_text()
    assert "_loader_codebook_flags([(False, True)] * 3)" in src
    assert "None, _mul1_marker(device)," in src


def test_case_b_runs_both_dispatch_families():
    tool = _tool()
    # m=1 exercises the small-m GEMV family the decode batch uses;
    # m=64 the reconstruct+hgemm prefill family. Run 1 only tested m=64.
    assert tool.B_MS == (1, 64)


def test_case_c_tp_probe_needs_no_vllm_layer():
    # The TP>1 rejection's preconditions resolve without constructing a
    # vLLM linear: _indexer_shards keys on the class NAME and
    # _resolve_tp_geometry on the declared tp_size. Run 1 subclassed
    # QKVParallelLinear and died with a TypeError in __init__, so the
    # rejection contract was never exercised at all.
    tool = _tool()

    class MinimaxM3QKVParallelLinearWithIndexer:  # noqa: N801
        tp_size = 2

    probe = MinimaxM3QKVParallelLinearWithIndexer()
    method = exl3.Exl3LinearMethod(
        exl3.Exl3Config.__new__(exl3.Exl3Config), bits=4
    )
    assert method._indexer_shards(probe, 5) == (3, 4)
    assert exl3._resolve_tp_geometry(probe) == (0, 2)


def test_case_e_skips_without_a_token_file():
    # No mounted token file -> an explicit SKIP receipt entry, never a
    # silent pass and never a crash.
    tool = _tool()
    res = tool._case_e(torch.device("cpu"), "/nonexistent-token-file")
    assert "skip" in res and "pass" not in res


def test_case_e_constants_and_fixture_are_pinned():
    tool = _tool()
    assert tool.E_REPO == "vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw"
    assert tool.E_REVISION == "687cbff2c210b7514d18c5f6a4d1db3417b316c4"
    assert tool.E_SHARD == "model-00001-of-00016.safetensors"
    assert tool.E_MS == (1, 64)
    fixture = json.loads(
        (REPO / "tests/data/minimax_m3_layer0_real_shards.json").read_text()
    )
    assert len(fixture) == 16
    assert all(v["file"] == tool.E_SHARD for v in fixture.values())
    # The pack's mixed per-shard K: q K4, k/v K5 (trellis last dim / 16).
    ks = {}
    for p in ("q", "k", "v"):
        t = fixture[
            f"language_model.model.layers.0.self_attn.{p}_proj.trellis"
        ]
        assert t["dtype"] == "I16"
        assert t["bytes"] == 2 * t["shape"][0] * t["shape"][1] * t["shape"][2]
        ks[p] = t["shape"][2] // 16
        mul1 = fixture[
            f"language_model.model.layers.0.self_attn.{p}_proj.mul1"
        ]
        assert mul1["dtype"] == "I32" and mul1["shape"] == []
    assert ks == {"q": 4, "k": 5, "v": 5}
    # The indexer payload/scale geometry: F8_E4M3 [out, 6144] + U8
    # block-32 scales [out, 192].
    for p, out_f in (("index_q", 512), ("index_k", 128)):
        w = fixture[f"language_model.model.layers.0.self_attn.{p}_proj.weight"]
        s = fixture[
            f"language_model.model.layers.0.self_attn.{p}_proj.weight_scale_inv"
        ]
        assert w["dtype"] == "F8_E4M3" and w["shape"] == [out_f, 6144]
        assert s["dtype"] == "U8" and s["shape"] == [out_f, 192]
    # The fixture must be byte-identical to the parent's census when the
    # census is on this box.
    census_path = Path("/root/work/minimax-m31/pack/census_tensors.json")
    if census_path.is_file():
        census = json.loads(census_path.read_text())
        for name, ent in fixture.items():
            assert census[name] == ent, name


def test_case_e_trellis_needs_the_loader_3d_layout():
    # Run-3's Case E died inside LinearEXL3 ("trellis must have dim = 3")
    # because the tool kept the fetched region flat. The fix routes the
    # region through the plugin loader's own helpers: the checkpoint's
    # stored shape IS the 3-D [in/16, out/16, 16*K] layout every loader
    # slot adopts, and the loader's gate runs BEFORE any kernel call.
    tool = _tool()
    entries = tool._fixture_entries()
    for p, out_f in zip(("q", "k", "v"), tool.B_OUTS[:3]):
        name = f"language_model.model.layers.0.self_attn.{p}_proj.trellis"
        ent = entries[name]
        flat = torch.zeros(ent["bytes"], dtype=torch.uint8)
        t = exl3.coerce_checkpoint_region(flat, torch.int16, ent["shape"])
        # 3-D trellis resolved from the fixture shapes, K as censused.
        assert exl3.assert_exl3_trellis_3d(t, tool.B_IN, out_f) == (
            ent["shape"][2] // 16
        )
        # The flat region and every wrong geometry are rejected.
        with pytest.raises(ValueError):
            exl3.assert_exl3_trellis_3d(flat.view(torch.int16), tool.B_IN, out_f)
        with pytest.raises(ValueError):
            exl3.assert_exl3_trellis_3d(t[:, : t.shape[1] // 2, :], tool.B_IN, out_f)
        with pytest.raises(ValueError):
            exl3.coerce_checkpoint_region(flat[:-1], torch.int16, ent["shape"])
    # The tool's Case E fetch goes through the loader helpers, not a
    # hand-rolled reshape.
    src = (REPO / "tools/gb10_minimax_m3_parity.py").read_text()
    assert "exl3.coerce_checkpoint_region(" in src
    assert "exl3.assert_exl3_trellis_3d(trellis, B_IN, out_f)" in src


def test_case_d_run3_nan_was_a_tool_value_cast():
    # torch's .to(uint8) on an E4M3 payload is a VALUE cast (1.0 -> byte
    # 1), not a bitcast (byte 0x38 = 56). Run-3's Case D staged the
    # draft's weights through that value cast, so every weight collapsed
    # to byte 0, the kernel emitted the zero vector, and
    # cosine(zeros, reference) printed nan. The sm_121 verdict on
    # FlashInferCutlass was never actually tested.
    one = torch.ones((), dtype=torch.float32).to(torch.float8_e4m3fn)
    assert int(one.view(torch.uint8).item()) == 56  # the bit pattern
    assert int(one.to(torch.uint8).item()) == 1  # the value-cast trap
    # The fixed tool must stage the fp8 payload as-is: no .to(torch.uint8)
    # anywhere in Case D (bit-level access for the reference decode goes
    # through view, the bitcast).
    src = (REPO / "tools/gb10_minimax_m3_parity.py").read_text()
    seg = src.split("def _case_d(", 1)[1].split("# ── Case E", 1)[0]
    assert ".to(torch.uint8)" not in seg


def test_case_d_sweeps_every_cuda_candidate():
    tool = _tool()
    # Pinned == vllm's _POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA]
    # (kernels/linear/__init__.py); the sweep cross-checks the live list
    # at runtime and refuses to run on a reshuffled candidate set.
    assert sorted(tool.D_CANDIDATES) == sorted({
        "FlashInferCutedslMxfp8LinearKernel",
        "FlashInferCutlassMxfp8LinearKernel",
        "MarlinMxfp8LinearKernel",
        "B12xMxfp8LinearKernel",
        "EmulationMxfp8LinearKernel",
        "HummingMxfp8LinearKernel",
        "FlashInferTrtllmMxfp8LinearKernel",
        "TorchMxFp8LinearKernel",
    })
    # The activation classification matches each kernel's apply: Marlin
    # is W8A16 and Emulation dequantizes to BF16 at load, so their
    # references use the raw activation; every other candidate quantizes
    # the activation in-kernel.
    assert "MarlinMxfp8LinearKernel" not in tool.D_ACT_QUANTIZING
    assert "EmulationMxfp8LinearKernel" not in tool.D_ACT_QUANTIZING
    assert tool.D_ACT_QUANTIZING <= set(tool.D_CANDIDATES)
    # The sanity payload (E4M3 0x38 under all-127 E8M0 scales) decodes to
    # exactly 1.0, so the sanity reference is the activation row sums.
    s = tool._mxfp8_dequant(
        torch.full((2, 64), 56, dtype=torch.uint8),
        torch.full((2, 2), 127, dtype=torch.uint8),
    )
    assert (s == 1.0).all()


def test_case_a_receipt_must_name_the_inner_dispatch_path():
    # Run 3's Case A printed path=fused: the coop act-3 fast path returns
    # early from inside the fused entry point, and the wrapper then
    # clobbered the diagnostic label with the generic entry name, so the
    # tool's path bar could never be satisfied and the receipt could not
    # say which kernel actually ran. The inner path must stamp the truth,
    # the wrapper must not clobber it, and the tool must require BOTH
    # the coop decode path and a correct prefill-family probe.
    esrc = (REPO / "src/vllm_exl3/exl3.py").read_text()
    coop_seg = esrc.split("exl3_moe_coop(", 1)[1].split("return out", 1)[0]
    assert 'layer._exl3_last_apply = "coop"' in coop_seg
    fused_idx = esrc.find('layer._exl3_last_apply = "fused"')
    assert fused_idx > 0
    assert "is None:" in esrc[fused_idx - 200:fused_idx]
    tool = _tool()
    src = (REPO / "tools/gb10_minimax_m3_parity.py").read_text()
    assert 'getattr(layer, "_exl3_last_apply", "unknown")' in src
    seg = src.split("def _case_a_run(", 1)[1].split("# ── Case B", 1)[0]
    assert 'path == "coop"' in seg
    assert "prefill_cos >= COS_MIN" in seg
    # The decode probe is inside the coop slot cap; the prefill probe is
    # past it (the expert-batched exl3_moe path prefill takes).
    assert tool.A_TOKENS * tool.TOPK <= 256
    assert tool.A_PREFILL_TOKENS * tool.TOPK > 256


# ── DSpark draft quantization: the plugin declines by construction ─────────


def _dspark_quant_card():
    """The exact quantization_config of pack/dspark/config.json."""
    path = Path("/root/work/minimax-m31/pack/dspark_config.json")
    if not path.is_file():
        pytest.skip("pack/dspark_config.json not on this box")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["quantization_config"]


def test_plugin_never_claims_the_dsparks_mxfp8_quant_method():
    # The draft checkpoint tags quant_method "mxfp8"; the plugin's override
    # only ever claims "exl3", so the draft resolves to vLLM's native
    # ModelOptMxFp8Config (registry maps "mxfp8" -> ModelOptMxFp8Config) and
    # the plugin is never the draft's quant config.
    card = _dspark_quant_card()
    assert card["quant_method"] == "mxfp8"
    assert card["weight_block_size"] == [1, 32]
    got = exl3.Exl3Config.override_quantization_method(
        card, user_quant=None, hf_config=None
    )
    assert got is None


def test_dsparks_rewritten_hf_config_is_not_the_minimax_target():
    # vLLM rewrites the draft hf_config to model_type "minimax_m3_dspark"
    # with architectures ["DSparkMiniMaxDraftModel"] (config/speculative.py).
    # The plugin's MiniMax coverage marker must not fire for it: the draft
    # would otherwise be handed EXL3 methods for MXFP8 tensors.
    draft = SimpleNamespace(
        model_type="minimax_m3_dspark",
        architectures=["DSparkMiniMaxDraftModel"],
    )
    assert exl3._hf_config_is_minimax_m3(draft) is False
    target = SimpleNamespace(
        model_type="minimax_m3_vl",
        architectures=["MiniMaxM3SparseForConditionalGeneration"],
    )
    assert exl3._hf_config_is_minimax_m3(target) is True


def test_dsparks_ignored_layers_are_not_exl3_families():
    # The ignored_layers entries the draft model actually instantiates are
    # its own heads (markov/confidence/fc); those must be outside the
    # plugin's MiniMax EXL3 coverage. The remaining entries are not draft
    # modules at all: embed/lm_head are shared with the target (dspark.py
    # docstring: the checkpoint ships neither), the block_sparse_moe.gate
    # entries are unquantized router weights, and the vision entries do not
    # exist in this text-only pack.
    card = _dspark_quant_card()
    ignored = card["ignored_layers"]
    instantiated = [
        n for n in ignored
        if ".dspark." in n and "block_sparse_moe" not in n
    ]
    assert instantiated  # the four heads are in the list
    for name in instantiated:
        for fam in exl3.MINIMAX_NON_ROUTED_FAMILIES:
            assert not name.endswith(fam), (name, fam)


# ── Provisional alias selection ────────────────────────────────────────────


def _synthetic_census(n_layers=3, n_experts=8):
    """Census-shaped dict: layers 3..3+n-1, per-expert w1/w2/w3 trellises with
    the pack's K-triple distribution scaled down."""
    census = {}
    triples = [(2, 2, 2), (2, 3, 2), (3, 3, 3)]
    for li in range(n_layers):
        layer = 3 + li
        for eid in range(n_experts):
            kg, kd, ku = triples[eid % len(triples)]
            for proj, k in (("w1", kg), ("w2", kd), ("w3", ku)):
                # bytes = in*out*K/16 elements * 2B, in=32 out=16 (scaled down)
                census[
                    f"language_model.model.layers.{layer}.block_sparse_moe."
                    f"experts.{eid}.{proj}.trellis"
                ] = {
                    "dtype": "I16",
                    "shape": [2, 1, 16 * k],
                    "bytes": 2 * 1 * 16 * k * 2,
                }
    return census


sys.path.insert(0, str(REPO / "tools"))
import minimax_provisional_alias as mpa  # noqa: E402

sys.path.pop(0)


def _select(census, budget_bytes):
    experts = mpa.parse_census(census)
    mpa.validate(experts, 3, 8)
    table = mpa.expert_table(experts)
    return mpa.select_provisional(table, budget_bytes)


def _layer_num(key: str) -> int:
    return int(str(key).rsplit(".", 1)[1])


def _chosen(sel):
    return {
        (int(str(l).rsplit(".", 1)[1]), e)
        for l, eids in sel["experts"].items() for e in eids
    }


def test_provisional_alias_spreads_evenly_and_prefers_k222():
    census = _synthetic_census()
    # Budget for ~6 experts total: 2 per layer.
    per_expert = sorted(
        mpa.expert_table(mpa.parse_census(census)), key=lambda c: c["bytes"]
    )
    budget = 6 * per_expert[0]["bytes"]
    sel = _select(census, budget)
    layers = {
        _layer_num(l): len(v) for l, v in sel["experts"].items()
    }
    assert sorted(layers) == [3, 4, 5]
    assert set(layers.values()) <= {2, 3}
    # Every chosen expert is a (2,2,2) expert (the smallest).
    table = {
        (c["layer"], c["eid"]): c for c in mpa.expert_table(mpa.parse_census(census))
    }
    for layer, eid in _chosen(sel):
        assert table[(layer, eid)]["k"] == (2, 2, 2)


def test_provisional_alias_is_deterministic_and_capped():
    census = _synthetic_census()
    total = sum(
        c["bytes"] for c in mpa.expert_table(mpa.parse_census(census))
    )
    sel1 = _select(census, total // 2)
    sel2 = _select(census, total // 2)
    assert sel1["experts"] == sel2["experts"]
    assert sel1["used_bytes"] <= total // 2
    # Full-total budget selects every expert; over-budget selection is exact.
    sel_all = _select(census, total)
    assert sel_all["used_bytes"] == total
    assert len(_chosen(sel_all)) == 24


def test_provisional_alias_rejects_bad_censuses():
    census = _synthetic_census()
    experts = mpa.parse_census(census)
    with pytest.raises(ValueError):  # wrong layer count
        mpa.validate(experts, 4, 8)
    with pytest.raises(ValueError):  # wrong expert count
        mpa.validate(experts, 3, 9)
    bad = dict(census)
    key = next(iter(bad))
    bad[key] = dict(bad[key], shape=[2, 1, 16 * 9])  # unsupported K
    with pytest.raises(ValueError):
        mpa.parse_census(bad)


def test_provisional_alias_output_is_ats_build_alias_compatible(tmp_path):
    # The selection file must be loadable by ats_build_alias.py --experts-json:
    # pure {"layers.N": [eids]} with integer ids, nothing else at top level.
    census = _synthetic_census()
    sel = _select(census, 1 << 20)
    data = json.loads(json.dumps(sel["experts"]))
    parsed = {
        str(k): sorted({int(x) for x in v}) for k, v in data.items() if v
    }
    assert parsed == {k: sorted(v) for k, v in parsed.items()}
    assert all(k.startswith("layers.") for k in parsed)


# ── Stats-based regeneration: the MiniMax expert family must resolve ───────


def test_ats_build_alias_regex_matches_the_minimax_family():
    sys.path.insert(0, str(REPO / "tools"))
    try:
        import ats_build_alias as aba
    finally:
        sys.path.pop(0)
    good = (
        "language_model.model.layers.3.block_sparse_moe.experts.5.w1.trellis"
    )
    legacy = "layers.38.ffn.experts.12.w2.trellis"
    bad = "language_model.model.layers.3.block_sparse_moe.experts.5.w1.suh"
    m = aba._TRELLIS_RE.search(good)
    assert m is not None and m.group(1) == "layers.3"
    assert m.group(2) == "block_sparse_moe"
    assert m.group(3) == "5" and m.group(4) == "w1"
    m = aba._TRELLIS_RE.search(legacy)
    assert m is not None and m.group(2) == "ffn" and m.group(3) == "12"
    assert aba._TRELLIS_RE.search(bad) is None


def test_select_cold_maps_rows_back_to_layers_not_mtp(tmp_path):
    # MiniMax recipe: ROUTE_STATS rows 0..59 are the target layers 0..59 and
    # there are no MTP rows; mtp_offset must be set >= 60 so every histogram
    # row maps back to layers.N (the default 40 would misread rows 40..59 as
    # mtp.0..19 and silently drop those layers from cold-set selection).
    import numpy as np

    sys.path.insert(0, str(REPO / "tools"))
    try:
        import ats_build_alias as aba
    finally:
        sys.path.pop(0)
    n_rows, n_exp = 60, 128
    counts = np.zeros((n_rows, n_exp), dtype=np.int64)
    rows = np.zeros((n_rows,), dtype=np.int64)
    rows[:] = 1000
    # Layer 3: heavily touched experts 0..99 (hot); 100..127 untouched.
    counts[3, :100] = 5
    # Layer 59: one touched expert; the rest untouched.
    counts[59, 11] = 5
    meta = np.array(json.dumps({"mtp_offset": 60}))
    npz = tmp_path / "route_stats.npz"
    np.savez(npz, counts_0=counts, rows_0=rows, meta=meta)
    expert_bytes = {
        ("layers.3", e): (1000, []) for e in range(n_exp)
    }
    expert_bytes.update({("layers.59", e): (1000, []) for e in range(n_exp)})
    # Enough budget to exhaust layer 3's cold supply and reach layer 59.
    sel = aba.select_cold(str(npz), 30000, expert_bytes, log=lambda *_: None)
    assert sorted(sel) == ["layers.3", "layers.59"]
    assert "mtp.19" not in sel and "mtp.0" not in sel
    # The cold set excludes the touched experts, on both layers.
    assert all(e >= 100 for e in sel["layers.3"])
    assert 11 not in sel["layers.59"]


def test_select_cold_default_offset_drops_high_layers(tmp_path):
    # Negative control: with the DEFAULT mtp_offset (40, the DSV4.1 layout),
    # histogram rows 40..59 are misread as mtp.0..19 and layer 59 silently
    # disappears from selection. This is why the MiniMax recipe pins
    # VLLM_EXL3_ROUTE_STATS_MTP_OFFSET=60.
    import numpy as np

    sys.path.insert(0, str(REPO / "tools"))
    try:
        import ats_build_alias as aba
    finally:
        sys.path.pop(0)
    n_rows, n_exp = 60, 128
    counts = np.zeros((n_rows, n_exp), dtype=np.int64)
    rows = np.zeros((n_rows,), dtype=np.int64)
    rows[:] = 1000
    counts[3, :100] = 5
    counts[59, 11] = 5
    meta = np.array(json.dumps({}))
    npz = tmp_path / "route_stats.npz"
    np.savez(npz, counts_0=counts, rows_0=rows, meta=meta)
    expert_bytes = {
        ("layers.3", e): (1000, []) for e in range(n_exp)
    }
    expert_bytes.update({("layers.59", e): (1000, []) for e in range(n_exp)})
    sel = aba.select_cold(str(npz), 30000, expert_bytes, log=lambda *_: None)
    assert sorted(sel) == ["layers.3"]


def test_route_stats_row_mapping_with_the_minimax_recipe(monkeypatch):
    # The documented MiniMax env recipe (ROWS=60, EXPERTS=128, MTP_OFFSET=60)
    # must map every routed layer name to its own histogram row.
    monkeypatch.setenv("VLLM_EXL3_ROUTE_STATS", "1")
    monkeypatch.setenv("VLLM_EXL3_ROUTE_STATS_DIR", "/tmp")
    monkeypatch.setenv("VLLM_EXL3_ROUTE_STATS_ROWS", "60")
    monkeypatch.setenv("VLLM_EXL3_ROUTE_STATS_EXPERTS", "128")
    monkeypatch.setenv("VLLM_EXL3_ROUTE_STATS_MTP_OFFSET", "60")
    from vllm_exl3 import route_stats

    rs = importlib.reload(route_stats)
    try:
        assert rs.enabled() is True  # latches the env recipe
        for layer in (3, 33, 59):
            fake = SimpleNamespace(layer_name=f"model.layers.{layer}.block_sparse_moe")
            assert rs._row_of(fake) == layer
        # Expert ids are < 128, so the remainder() guard is the identity for
        # real routes (the histogram is 60x128, not 48x384).
        assert rs._N_ROWS == 60 and rs._N_EXP == 128
        assert rs._MTP_OFFSET == 60
    finally:
        monkeypatch.delenv("VLLM_EXL3_ROUTE_STATS")
        importlib.reload(route_stats)  # restore defaults (module is env-latched)
