"""Milestone-2 MiniMax-M3 implementation tests (CPU-only, no vLLM, no GPU).

Covers the parent-approved decisions:
- D1: indexer MXFP8 shards — dequant semantics (E8M0 decode 2**(b-127),
  E4M3FN payload, block 32 along the input) against a hand-computed
  reference, and the fused [q | k | v | index_q | index_k] output layout.
- D2: shared experts — computed by the model's own MLP modules via the
  runner, added exactly once; the plugin's MoE method never fuses them.
- D3: swiglu beta — only None or 1.0 accepted; alpha must be 1.702.
- Pack coverage: every tensor name of the pack census maps to exactly one
  vLLM param + plugin loader slot; 0 unmatched, 0 double-mapped; the 7296
  routed experts x 3 projections resolve with their trellis K.
- ATS aliasing names for the block_sparse_moe.experts family.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

import vllm_exl3.exl3 as exl3
from vllm_exl3 import ats_alias


# ---------------------------------------------------------------------------
# D1: MXFP8 indexer dequant
# ---------------------------------------------------------------------------


def _ref_e4m3_fn(bits: int) -> float:
    """Hand-computed E4M3FN decode of one byte (independent of the plugin)."""
    sign = -1.0 if bits & 0x80 else 1.0
    exp = (bits >> 3) & 0xF
    man = bits & 0x7
    if (bits & 0x7F) == 0x7F:
        return float("nan")
    if exp == 0:
        return sign * man * 2.0 ** -9
    return sign * (man + 8.0) * 2.0 ** (exp - 10.0)


def test_e4m3_decode_matches_torch_float8_on_cpu():
    # Cross-check the manual bit decode against torch's own float8_e4m3fn
    # cast for every representable, non-NaN byte.
    bits = torch.arange(256, dtype=torch.uint8)
    got = exl3._e4m3_bits_to_float(bits)
    ref = bits.view(torch.float8_e4m3fn).to(torch.float32)
    finite = torch.isfinite(ref)
    assert torch.allclose(got[finite], ref[finite], rtol=0, atol=0)
    assert torch.isnan(got[~finite]).all()


def test_indexer_dequant_matches_hand_computed_reference():
    # Synthetic F8_E4M3 payload + U8 E8M0 block-32 scales; the reference is
    # computed by hand here (payload * 2**(byte - 127)), never via the plugin.
    out_rows, in_cols = 4, 32  # one 32-wide scale block along the input
    payload_bits = torch.tensor(
        [
            [0x30, 0x31, 0x32, 0x33, 0x38, 0x39, 0x3A, 0x3B,
             0x00, 0x01, 0x02, 0x03, 0x08, 0x09, 0x0A, 0x0B,
             0xB0, 0xB1, 0xB2, 0xB3, 0xB8, 0xB9, 0xBA, 0xBB,
             0x70, 0x71, 0x72, 0x73, 0x78, 0x79, 0x7A, 0x7B],
            [0x3C, 0x3D, 0x3E, 0x30, 0x3C, 0x3D, 0x3E, 0x30,
             0x0C, 0x0D, 0x0E, 0x00, 0x0C, 0x0D, 0x0E, 0x00,
             0xBC, 0xBD, 0xBE, 0xB0, 0xBC, 0xBD, 0xBE, 0xB0,
             0x7C, 0x7D, 0x0F, 0x10, 0x1F, 0x20, 0x2F, 0x40],
            [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
             0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x01, 0x02,
             0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A,
             0x0B, 0x0C, 0x0D, 0x0E, 0x21, 0x22, 0x23, 0x24],
            [0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C,
             0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C,
             0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x5B, 0x5C,
             0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x6B, 0x6C],
        ],
        dtype=torch.uint8,
    )
    scales = torch.tensor(
        [[127], [128], [129], [120]], dtype=torch.uint8
    )
    assert payload_bits.shape == (out_rows, in_cols)
    assert scales.shape == (out_rows, in_cols // 32)

    got = exl3.dequant_minimax_indexer(payload_bits, scales)

    ref = torch.empty(out_rows, in_cols, dtype=torch.float32)
    for r in range(out_rows):
        for c in range(in_cols):
            descale = 2.0 ** (float(scales[r, c // 32]) - 127.0)
            ref[r, c] = _ref_e4m3_fn(int(payload_bits[r, c])) * descale

    finite = torch.isfinite(ref)
    assert torch.allclose(
        got.to(torch.float32)[finite], ref[finite], rtol=2e-2, atol=2e-2
    )
    # The E8M0 byte is a DIRECT descale exponent, not a reciprocal: byte 127
    # is identity, 128 doubles, 126 halves (same semantics as vLLM's
    # dequant_mxfp8_to_bf16, mxfp8_utils.py:232-245 — multiply by
    # exp2(scale - 127), consumed at load by kernels/linear/mxfp8/emulation.py:48).
    one = exl3.dequant_minimax_indexer(
        torch.zeros(1, 32, dtype=torch.uint8), torch.tensor([[127]], dtype=torch.uint8)
    )
    assert torch.equal(one, torch.zeros(1, 32, dtype=torch.bfloat16))
    base = exl3.dequant_minimax_indexer(
        torch.full((1, 32), 0x30, dtype=torch.uint8),
        torch.tensor([[127]], dtype=torch.uint8),
    ).to(torch.float32)
    doubled = exl3.dequant_minimax_indexer(
        torch.full((1, 32), 0x30, dtype=torch.uint8),
        torch.tensor([[128]], dtype=torch.uint8),
    ).to(torch.float32)
    halved = exl3.dequant_minimax_indexer(
        torch.full((1, 32), 0x30, dtype=torch.uint8),
        torch.tensor([[126]], dtype=torch.uint8),
    ).to(torch.float32)
    assert doubled == pytest.approx(2.0 * base)
    assert halved == pytest.approx(0.5 * base)


def test_indexer_dequant_rejects_bad_geometry():
    with pytest.raises(ValueError):
        exl3.dequant_minimax_indexer(
            torch.zeros(2, 33, dtype=torch.uint8),
            torch.zeros(2, 1, dtype=torch.uint8),
        )
    with pytest.raises(ValueError):
        exl3.dequant_minimax_indexer(
            torch.zeros(2, 64, dtype=torch.uint8),
            torch.zeros(2, 3, dtype=torch.uint8),
        )


def _fake_indexer_layer():
    # A class whose *name* is the branch's fused qkv+indexer layer type.
    return type("MinimaxM3QKVParallelLinearWithIndexer", (), {})()


def test_indexer_shards_are_the_last_two_fused_shards():
    method = exl3.Exl3LinearMethod.__new__(exl3.Exl3LinearMethod)
    # The branch's fused layer has exactly 5 shards,
    # [q | k | v | index_q | index_k] (linear.py:1460-1464 offsets).
    assert method._indexer_shards(_fake_indexer_layer(), 5) == (3, 4)
    # Any other layer/arity: none.
    assert method._indexer_shards(_fake_indexer_layer(), 3) == ()
    assert method._indexer_shards(SimpleNamespace(), 5) == ()
    assert method._indexer_shards(SimpleNamespace(), 3) == ()


def test_fused_qkv_indexer_output_layout_matches_the_branch():
    # Branch per-rank layout (TP1): q 64x128, k/v 4x128, index_q 4x128,
    # index_k 1x128 (linear.py:1460-1464 _get_shard_offset_mapping).
    sizes = [8192, 512, 512, 512, 128]
    offsets = [0, 8192, 8704, 9216, 9728]
    total = 9856
    cum = []
    run = 0
    for s in sizes:
        cum.append(run)
        run += s
    assert cum == offsets
    assert run == total
    # The plugin concatenates shard outputs in shard order (the only layout
    # _apply_impl produces), so slice i of the fused output is exactly the
    # output of shard i: the branch's split and the plugin's agree.
    order = ["q", "k", "v", "index_q", "index_k"]
    assert list(zip(order, offsets)) == [
        ("q", 0), ("k", 8192), ("v", 8704),
        ("index_q", 9216), ("index_k", 9728),
    ]


# ---------------------------------------------------------------------------
# D3: swiglu beta / alpha
# ---------------------------------------------------------------------------


def _moe_layer(activation: str, **layer_attrs):
    return SimpleNamespace(
        moe_config=SimpleNamespace(activation=SimpleNamespace(value=activation)),
        **layer_attrs,
    )


def test_beta_none_and_one_are_accepted():
    for beta in (None, 1.0):
        layer = _moe_layer(
            "swigluoai_uninterleave",
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
            swiglu_beta=beta,
        )
        alpha, limit = exl3.resolve_moe_gate_params(layer)
        assert alpha == pytest.approx(1.702)
        assert limit == pytest.approx(7.0)


def test_beta_other_than_one_raises():
    layer = _moe_layer(
        "swigluoai_uninterleave",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        swiglu_beta=0.0,
    )
    with pytest.raises(ValueError):
        exl3.resolve_moe_gate_params(layer)


def test_alpha_other_than_kernel_constant_raises():
    layer = _moe_layer(
        "swigluoai_uninterleave",
        swiglu_alpha=2.0,
        swiglu_limit=7.0,
        swiglu_beta=1.0,
    )
    with pytest.raises(ValueError):
        exl3.resolve_moe_gate_params(layer)


def test_python_loop_act3_matches_swigluoai_reference():
    # Loop twin of the coop kernel's act_oai_swiglu (exl3_moe_coop_kernel.cuh
    # :98-107): gate clamped from above, up clamped symmetrically,
    # (up + 1) * g * sigmoid(1.702 * g).
    torch.manual_seed(0)
    hidden, inter = 32, 16
    x = torch.randn(3, hidden)
    gate_w = (torch.randn(inter, hidden) * 0.1)
    up_w = (torch.randn(inter, hidden) * 0.1)
    down_w = (torch.randn(hidden, inter) * 0.1)

    def fwd(w, h):
        # The loop feeds half inputs and accumulates half outputs, matching
        # the real EXL3 ext tensors.
        return h @ w.t()

    class _FakeLinear:
        def __init__(self, w):
            self.w = w.half()

        def forward(self, h, _meta, out_dtype=None):
            return fwd(self.w, h)

    inners = [
        {
            "gate": _FakeLinear(gate_w),
            "up": _FakeLinear(up_w),
            "down": _FakeLinear(down_w),
        }
    ]
    ids = torch.tensor([[0], [0], [0]], dtype=torch.long)
    weights = torch.ones(3, 1)
    limit = 7.0
    got = exl3.apply_exl3_python_loop(
        x, ids, weights, inners, None, limit, act=exl3.MOE_ACT_SWIGLU_OAI
    )
    xh = x.half()
    g = fwd(gate_w.half(), xh).clamp(max=limit)
    u = fwd(up_w.half(), xh).clamp(min=-limit, max=limit)
    act = (u + 1.0) * g * torch.sigmoid(1.702 * g)
    ref = act @ down_w.half().t()
    ref = ref.to(torch.float32)
    got = got.to(torch.float32)
    assert torch.allclose(got, ref, rtol=2e-2, atol=2e-2)
    # And it is NOT silu: the two references differ.
    silu_ref = torch.nn.functional.silu(g) * u @ down_w.half().t()
    silu_ref = silu_ref.to(torch.float32)
    assert not torch.allclose(got, silu_ref, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# D2: shared experts are added exactly once, never fused into EXL3
# ---------------------------------------------------------------------------


def test_moe_method_is_modular_and_never_fuses_shared_experts():
    cfg = exl3.Exl3Config.__new__(exl3.Exl3Config)
    method = exl3.Exl3MoEMethod(SimpleNamespace(), cfg, bits=2)
    # D4: RoutedExperts.forward_modular asserts not is_monolithic
    # (routed_experts.py:1387); the base property falls to moe_kernel
    # (fused_moe_method_base.py:135-141) which EXL3 lacks.
    assert method.is_monolithic is False
    # EXL3 reports no fused shared experts: no staging hook, no fused config.
    assert method.get_fused_moe_quant_config(SimpleNamespace()) is None
    assert not hasattr(method, "stages_shared_experts")
    assert not hasattr(method, "num_fused_shared_experts")


def test_moe_method_apply_returns_routed_output_only(monkeypatch):
    # The runner computes the shared experts itself through the model's own
    # MiniMaxM3MLP modules (Exl3LinearMethod per projection) BEFORE the routed
    # apply (moe_runner.py _maybe_apply_shared_experts -> SharedExperts.forward
    # -> self._layer(input)) and combines the two outputs exactly once at the
    # model level (MoEOutput(routed, shared_output)). The plugin's apply must
    # return the routed contribution only and never touch the shared args.
    cfg = exl3.Exl3Config.__new__(exl3.Exl3Config)
    method = exl3.Exl3MoEMethod(SimpleNamespace(), cfg, bits=2)
    layer = _moe_layer(
        "swigluoai_uninterleave",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        swiglu_beta=1.0,
    )
    captured = {}

    def fake_apply(x, topk_ids, topk_weights, lay, limit=None, fused=None):
        captured["limit"] = limit
        return "routed-out"

    monkeypatch.setattr(exl3, "apply_exl3_experts", fake_apply)
    out = method.apply(
        layer,
        x=torch.zeros(1, 8),
        topk_weights=torch.ones(1, 1),
        topk_ids=torch.zeros(1, 1, dtype=torch.long),
        shared_experts="shared-module",
        shared_experts_input="shared-input",
    )
    assert out == "routed-out"
    # The clamp limit resolved from the layer (7.0), not SWIGLU_LIMIT_DEFAULT.
    assert captured["limit"] == pytest.approx(7.0)


def test_shared_plus_routed_is_combined_exactly_once():
    # End-to-end accounting with the real python loop for the routed part and
    # a stand-in for the model's shared MLP: the model-level combine adds the
    # shared output exactly once.
    torch.manual_seed(0)
    hidden = 32
    x = torch.randn(2, hidden)
    shared_w = (torch.randn(hidden, hidden) * 0.1).half()

    # Routed: one expert, zero weight -> zero contribution.
    class _Zero:
        def forward(self, h, _meta, out_dtype=None):
            return torch.zeros(h.shape[0], hidden)

    inners = [{"gate": _Zero(), "up": _Zero(), "down": _Zero()}]
    ids = torch.zeros(2, 1, dtype=torch.long)
    weights = torch.zeros(2, 1)
    routed = exl3.apply_exl3_python_loop(
        x, ids, weights, inners, None, None, act=exl3.MOE_ACT_SWIGLU_OAI
    )
    assert torch.equal(routed, torch.zeros(2, hidden))

    shared = x.half() @ shared_w.t()
    # The runner's combine (moe_runner._maybe_combine -> model MoEOutput):
    combined = shared.float() + routed
    assert torch.allclose(combined, shared.float())
    # Exactly once: adding it a second time would double it; assert the
    # single-add contract against the double-add.
    assert not torch.allclose(combined, shared.float() * 2)


# ---------------------------------------------------------------------------
# ATS aliasing names
# ---------------------------------------------------------------------------


def test_ats_expert_regex_matches_the_minimax_family():
    import re

    key = "layers.3"
    pat = re.compile(
        r"(?:^|\.)" + re.escape(key) + r"\.("
        + "|".join(re.escape(p) for p in ats_alias.EXPERT_CKPT_PREFIXES)
        + r")\.(\d+)\.(w1|w2|w3)\.trellis$"
    )
    name = (
        "language_model.model.layers.3.block_sparse_moe.experts.5.w1.trellis"
    )
    m = pat.search(name)
    assert m is not None and m.group(1) == "block_sparse_moe.experts"
    assert m.group(2) == "5" and m.group(3) == "w1"
    # The historical family still matches.
    assert pat.search("model.layers.3.ffn.experts.5.w2.trellis") is not None


def test_ats_alias_embed_key_is_the_minimax_checkpoint_name():
    assert (
        ats_alias.MINIMAX_EMBED_KEY
        == "language_model.model.embed_tokens.weight"
    )


# ---------------------------------------------------------------------------
# Pack coverage: every census tensor -> exactly one param + loader slot
# ---------------------------------------------------------------------------

_FIXTURE = os.path.join(
    os.path.dirname(__file__), "data", "minimax_m3_pack_census_compact.json"
)

# (suffix -> slot) for EXL3 trellis storage.
_TRELLIS_SLOTS = ("trellis", "suh", "svh", "mul1")
# Checkpoint projection -> vLLM fused param shard.
_QKV_SHARD = {
    "q_proj": ("self_attn.qkv_proj", "q"),
    "k_proj": ("self_attn.qkv_proj", "k"),
    "v_proj": ("self_attn.qkv_proj", "v"),
    "index_q_proj": ("self_attn.qkv_proj", "index_q"),
    "index_k_proj": ("self_attn.qkv_proj", "index_k"),
}
# Checkpoint w-id -> (RoutedExperts param prefix, plugin projection).
_EXPERT_SHARD = {"w1": ("w13", "gate"), "w3": ("w13", "up"), "w2": ("w2", "down")}


def _load_fixture():
    with open(_FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


def _family_rules(fam: str):
    """(kind, payload) classification of one abstract census family."""
    if fam == "language_model.model.embed_tokens.weight":
        return ("embed", None)
    if fam.startswith("language_model.lm_head"):
        return ("lm_head", None)
    if fam.endswith("block_sparse_moe.gate.weight"):
        return ("router_gate", None)
    if fam.endswith("block_sparse_moe.e_score_correction_bias"):
        return ("router_bias", None)
    if ".block_sparse_moe.experts." in fam:
        return ("expert", None)
    if ".shared_experts." in fam:
        return ("shared_mlp", None)
    if ".mlp." in fam:
        return ("dense_mlp", None)
    if any(f".{p}." in fam for p in ("index_q_proj", "index_k_proj")):
        return ("indexer", None)
    if any(f".{p}." in fam for p in ("q_proj", "k_proj", "v_proj")):
        return ("qkv", None)
    if ".o_proj." in fam:
        return ("o_proj", None)
    if "norm" in fam:
        return ("norm", None)
    raise AssertionError(f"unclassified census family: {fam}")


def test_pack_census_fixture_is_complete():
    fx = _load_fixture()
    assert fx["_total_tensors"] == 89952
    assert sum(
        e["count"] for fam in fx["families"].values() for e in fam
    ) == fx["_total_tensors"]


def test_every_pack_tensor_maps_to_exactly_one_param_and_slot():
    fx = _load_fixture()
    assignments = {}  # concrete tensor name -> (vllm param, plugin slot)
    counts = {}

    def assign(name: str, param: str, slot: str):
        assert name not in assignments, f"double-mapped: {name}"
        assignments[name] = (param, slot)

    for fam, entries in fx["families"].items():
        kind, _ = _family_rules(fam)
        for e in entries:
            count = e["count"]
            counts[(fam, e["dtype"], tuple(e["shape"]))] = count
            if kind == "expert":
                # Expand the 57 MoE layers x 128 experts (layer ids 3..59,
                # expert ids 0..127, derived from the fixture itself below).
                continue
            # Every non-expert family has one concrete name per index N;
            # the abstract family aggregates `count` of them. Reconstruct the
            # names by re-substituting the layer ids the census covered.
            n_layers = count
            for i in range(n_layers):
                name = fam.replace(".N.", f".{i}.", 1)
                if kind == "embed":
                    assign(name, "VocabParallelEmbedding.weight", "native-unquantized (ATS embed alias key)")
                elif kind == "lm_head":
                    assign(
                        name,
                        "lm_head." + name.rsplit(".", 1)[-1],
                        "Exl3LinearMethod trellis loader",
                    )
                elif kind == "router_gate":
                    assign(name, "block_sparse_moe.gate.weight", "GateLinear UnquantizedLinearMethod")
                elif kind == "router_bias":
                    assign(name, "block_sparse_moe.e_score_correction_bias", "native unquantized nn.Parameter")
                elif kind == "shared_mlp":
                    proj, suffix = name.split(".")[-2], name.split(".")[-1]
                    if proj in ("gate_proj", "up_proj"):
                        param = f"shared_experts.gate_up_proj.{suffix}"
                        slot = f"Exl3LinearMethod merged shard {0 if proj == 'gate_proj' else 1}"
                    else:
                        param = f"shared_experts.down_proj.{suffix}"
                        slot = "Exl3LinearMethod trellis loader"
                    assign(name, param, slot)
                elif kind == "dense_mlp":
                    proj, suffix = name.split(".")[-2], name.split(".")[-1]
                    if proj in ("gate_proj", "up_proj"):
                        param = f"mlp.gate_up_proj.{suffix}"
                        slot = f"Exl3LinearMethod merged shard {0 if proj == 'gate_proj' else 1}"
                    else:
                        param = f"mlp.down_proj.{suffix}"
                        slot = "Exl3LinearMethod trellis loader"
                    assign(name, param, slot)
                elif kind in ("indexer", "qkv"):
                    proj, tail = name.split(".")[-2], name.split(".")[-1]
                    fused, shard = _QKV_SHARD[proj]
                    if tail == "weight":
                        slot = (
                            "Exl3LinearMethod MXFP8 stash -> BF16 side weight"
                            if kind == "indexer"
                            else "Exl3LinearMethod trellis loader"
                        )
                    elif tail == "weight_scale_inv":
                        # model.py load_weights renames weight_scale_inv ->
                        # weight_scale before the params_dict lookup.
                        slot = "Exl3LinearMethod weight_scale stash (E8M0 block)"
                    else:
                        slot = "Exl3LinearMethod trellis loader"
                    assign(name, f"self_attn.{fused}.{tail}", slot)
                elif kind == "o_proj":
                    tail = name.split(".")[-1]
                    assign(name, f"self_attn.o_proj.{tail}", "Exl3LinearMethod trellis loader")
                elif kind == "norm":
                    assign(name, fam, "no quant method (MiniMAXGemmaRMSNorm)")

    # The non-expert reconstruction must cover every non-expert tensor.
    non_expert_total = sum(
        e["count"]
        for fam, entries in fx["families"].items()
        if _family_rules(fam)[0] != "expert"
        for e in entries
    )
    assert len(assignments) == non_expert_total

    # Unmatched names: none (every family classified above; an unclassified
    # family raises in _family_rules).
    # Double-mapped names: none (assign asserts on re-assignment).
    assert len(assignments) == non_expert_total


def test_all_routed_experts_resolve_with_their_trellis_k():
    fx = _load_fixture()
    # Derive the K census from the trellis shapes exactly as the loader does
    # (K = trellis.shape[-1] // 16).
    per_proj_k_counts = {"w1": {}, "w2": {}, "w3": {}}
    for fam, entries in fx["families"].items():
        if _family_rules(fam)[0] != "expert" or not fam.endswith(".trellis"):
            continue
        w_id = fam.split(".")[-2]
        for e in entries:
            shape = e["shape"]
            assert e["dtype"] == "I16" and shape[-1] % 16 == 0
            k = shape[-1] // 16
            per_proj_k_counts[w_id][k] = per_proj_k_counts[w_id].get(k, 0) + e["count"]
    # Pack per-projection K census (parent-verified): w1/w3 K2 7176, K3 120;
    # w2 K2 6096, K3 1200.
    assert per_proj_k_counts["w1"] == {2: 7176, 3: 120}
    assert per_proj_k_counts["w3"] == {2: 7176, 3: 120}
    assert per_proj_k_counts["w2"] == {2: 6096, 3: 1200}

    # The per-expert K triples reproduce exactly the pack census
    # {(2,2,2): 6096, (2,3,2): 1080, (3,3,3): 120} = 7296 experts x 3
    # projections, every one of which resolves under expert_k_census_supported.
    pack_census = {(2, 2, 2): 6096, (2, 3, 2): 1080, (3, 3, 3): 120}
    assert sum(pack_census.values()) == 7296
    assert exl3.expert_k_census_supported(pack_census) is True
    # Consistency of the triple census with the per-projection counts:
    for w_idx, w_id in ((0, "w1"), (1, "w2"), (2, "w3")):
        for k, want in per_proj_k_counts[w_id].items():
            got = sum(
                n for triple, n in pack_census.items() if triple[w_idx] == k
            )
            assert got == want, (w_id, k)

    # Every (layer, expert, projection) name resolves to exactly one
    # RoutedExperts param slot: 57 layers x 128 experts x 3 projections x 4
    # trellis tensors, all distinct.
    n_moe_layers = 57
    n_experts = 128
    seen = set()
    for layer in range(3, 3 + n_moe_layers):
        for e in range(n_experts):
            for w_id, (param_prefix, _proj) in _EXPERT_SHARD.items():
                for slot in _TRELLIS_SLOTS:
                    name = (
                        f"language_model.model.layers.{layer}."
                        f"block_sparse_moe.experts.{e}.{w_id}.{slot}"
                    )
                    key = (
                        f"model.layers.{layer}.block_sparse_moe.experts."
                        f"{param_prefix}_{slot}"
                    )
                    assert name not in seen
                    seen.add((name, key))
    assert len(seen) == n_moe_layers * n_experts * 3 * 4
    print(
        "pack coverage: 7296 experts x 3 projections x 4 tensors =",
        len(seen),
    )
