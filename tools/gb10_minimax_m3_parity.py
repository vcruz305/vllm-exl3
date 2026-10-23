#!/usr/bin/env python3
"""GB10 MiniMax-M3 EXL3 plugin parity probe (GPU_JOB-2).

Verifies, on the DGX Spark (GB10, sm_121), inside the dev image with this
worktree bind-mounted at /plugin and PYTHONPATH=/plugin/src:

  Case A  the production MoE entry point -- Exl3MoEMethod.apply on a layer
          built exactly as the loader leaves it -- resolves the MiniMax-M3
          act-3 (swiglu_oai) COOP kernel and matches a per-expert
          exl3_gemv reference with clamped SwiGLU-OAI, at K in {2, 3};
          the prefill-family probe (>256 slots) takes the expert-batched
          exl3_moe path and must match the same reference there;
  Case A2 the MIXED-K paths the real pack's serve actually runs: the three
          per-expert (K_gate, K_up, K_down) tuples of the pack's layer-3
          census, through the default unified exl3_moe_mixedk dispatch and
          the VLLM_EXL3_MIXEDK_MODE=grouped per-K-group dispatch, at decode
          and prefill sizes; every inner path must stamp its own label;
  Case B  the fused [q | k | v | index_q | index_k] linear: q/k/v EXL3
          trellis GEMMs plus the two MXFP8 indexer shards dequantized at
          load into BF16 side rows, sliced at the branch offsets, matching
          a per-shard reference in the branch's order, at m in {1, 64}
          (the m=1 arm exercises the small-m GEMV family the decode batch
          uses, the m=64 arm the reconstruct+hgemm prefill family);
  Case C  the rejection contract (CPU-side): wrong alpha/beta/activation
          raise, TP>1 indexer shards raise; no kernel launches;
  Case D  native-vLLM MXFP8 linear (the DSpark draft's format) at
          6144->12288 with block-32 E8M0 scales: EVERY selector candidate
          that claims support on this device runs through the exact
          method path the draft's ModelOpt method drives (staging ->
          process_weights_after_loading -> apply), against a per-family
          dequant reference plus an all-ones-scales/small-weights sanity
          payload; the receipt names the backend(s) that are correct and
          the auto selection.
  Case E  REAL pack bytes: layer 0's fused qkv+indexer tensors (q K4,
          k/v K5 EXL3 trellis + mul1 markers; index_q/index_k MXFP8)
          range-read from the Hub by authenticated HTTP with a token from
          --hf-token-file, run through the same fused layer, each slice
          compared against a definitional per-tensor dequant reference.
  Case F  REAL pack routed experts: three layer-3 experts -- one per
          (K_gate, K_up, K_down) tuple of the pack's census -- range-read
          from the Hub, run router-free (each row forced onto one expert)
          through the plugin's MoE apply at m in {1, 96}, i.e. exactly the
          mixed-K batched dispatch the serve takes for this pack at EVERY
          batch shape, against a definitional mul1-dequant + act-3
          reference per projection.

``--cpu-selftest`` runs everything that needs no kernel launch (Case C in
full, the pack/reference construction of A/B/D, the entry-point act
resolution through the real apply() with matmul-stub experts, and the
hand E4M3/E8M0 decode) on any box; the GPU arms run only with CUDA.

    python3 tools/gb10_minimax_m3_parity.py --cpu-selftest
    python3 tools/gb10_minimax_m3_parity.py --json receipt.json \
        --hf-token-file /run/hf_token

Case E is skipped (not failed) when no token file is mounted; every
Case E download failure or header/census mismatch is a failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Any

import torch

sys.path.insert(0, "src")

import vllm_exl3.exl3 as exl3  # noqa: E402

COS_MIN = 0.999
MEM_BUDGET_GIB = 4.0
TOPK = 2

# Case A shapes (GPU_JOB-2): 8 synthetic experts, hidden 4096, inter 1024,
# K in {2, 3} (the pack's per-expert K census), 64 tokens, topk 2.
A_HIDDEN, A_INTER, A_NEXP, A_TOKENS = 4096, 1024, 8, 64
# Prefill-family probe: tokens*topk = 384 > the coop kernel's 256-slot
# cap, so the dispatch takes the expert-batched exl3_moe path the serve
# takes at prefill sizes (real prefill slices into <=2048-row chunks of
# the same path). Below the fat threshold so no fat-GEMM route appears.
A_PREFILL_TOKENS = 192

# Case B shapes: the branch's fused layer at TP1 (hidden 6144; q 8192,
# k 512, v 512, index_q 512, index_k 128). Synthetic packs ~30 MB total.
B_IN = 6144
B_OUTS = (8192, 512, 512, 512, 128)
B_OFFSETS = (0, 8192, 8704, 9216, 9728)  # vllm linear.py fused split order
B_K = 4  # synthetic EXL3 K for q/k/v (the pack's q/k/v are K4/K5)
B_MS = (1, 64)  # decode-family and prefill-family row counts

# Case D shapes: the DSpark draft's dense FFN projection (hidden 6144 ->
# dspark_ffn_hidden_size 12288), MXFP8 block [1, 32] E8M0 scales.
D_N, D_K, D_TOKENS = 12288, 6144, 64

# Case E: layer 0 of the real pack. The 16 tensors live in shard 1; their
# census digest is pinned in tests/data/minimax_m3_layer0_real_shards.json.
E_REPO = "vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw"
E_REVISION = "687cbff2c210b7514d18c5f6a4d1db3417b316c4"
E_SHARD = "model-00001-of-00016.safetensors"
E_TOKEN_FILE = "/run/hf_token"
E_MS = (1, 64)

# Case A2 (mixed-K): the real pack's per-expert (K_gate, K_up, K_down)
# tuples -- layer 3 of the census, w1->gate / w3->up / w2->down -- with two
# synthetic experts per tuple, at the decode and prefill batch sizes the
# serve runs. Mixed-K layers never take the coop kernel (compat is False
# whenever the K groups are distinct), so both sizes exercise the batched
# mixed-K dispatch.
A_MIXED_TUPLES = ((2, 2, 2), (2, 2, 3), (3, 3, 3))
A_MIXED_MS = (64, 512)

# Case F (real pack routed experts): the layer-3 experts carrying the three
# distinct K tuples (census ground truth: expert 2 -> (2,2,2), expert 0 ->
# (2,2,3), expert 1 -> (3,3,3)), evaluated router-free at m in {1, 96}.
F_LAYER = 3
F_EXPERTS = (2, 0, 1)
F_HIDDEN, F_INTER = 6144, 3072
F_MS = (1, 96)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            a.reshape(-1).float(), b.reshape(-1).float(), dim=0
        )
    )


def _pack(in_f: int, out_f: int, k: int, device: torch.device) -> dict:
    """One synthetic EXL3 pack in the native fixture layout (int16 trellis +
    fp16 suh/svh), the layout gb10_exl3_moe_parity.py proved against the
    fused kernel."""
    return {
        "trellis": torch.randint(
            -32768, 32767, (in_f // 16, out_f // 16, 16 * k),
            dtype=torch.int16, device=device,
        ),
        "suh": (torch.randn(in_f, dtype=torch.float16, device=device) / 64.0),
        "svh": torch.randn(out_f, dtype=torch.float16, device=device),
    }


def _mul1_marker(device: torch.device) -> torch.Tensor:
    """The loader's per-shard mul1 codebook marker (the value the loader
    validates at process_weights_after_loading time; every EXL3 tensor of
    this pack ships one -- the census lists a `.mul1` I32 scalar next to
    every trellis/suh/svh triple)."""
    return torch.tensor(
        [exl3.MUL1_MARKER_SIGNED_INT32], dtype=torch.int32, device=device
    )


def _loader_codebook_flags(projections: list[tuple[bool, bool]]) -> tuple:
    """The loader's post-load (mcg, mul1) launch flags for one MoE layer,
    one flag pair per projection in [gate, up, down] order (exl3.py
    `_flag_pack`: the flags come from the loaded inners, and every expert
    must agree). Projections here are (has_mcg, has_mul1) pairs."""
    flags = tuple(
        b for pair in projections for b in pair
    )
    if len(flags) != 6:
        raise ValueError("expected three (mcg, mul1) projection pairs")
    return flags


def _gemv_ref(
    x: torch.Tensor, pack: dict, k: int, mcg: bool = False
) -> torch.Tensor:
    """Per-expert dequant GEMM reference (the proven native-map entry).

    ``mcg`` selects the codebook the trellis is decoded with, exactly as the
    production launch does: the pack's EXL3 tensors carry the mul1 marker,
    so the reference must pass mcg=False (the native entry maps
    mcg=False to the mul1 codebook). Passing mcg=True decodes a DIFFERENT
    codebook and is uncorrelated with the layer output by construction.
    """
    import vllm_exl3_c

    # The native GEMV accepts 1 <= m <= 8 rows per launch: run the reference in 8-row chunks.
    squeeze = x.dim() == 1
    rows = x.unsqueeze(0) if squeeze else x
    outs = [
        vllm_exl3_c.exl3_gemv(
            rows[i : i + 8].contiguous(), pack["trellis"], pack["suh"],
            pack["svh"], k, mcg, True,
        )
        for i in range(0, rows.shape[0], 8)
    ]
    out = torch.cat(outs, dim=0)
    return out.squeeze(0) if squeeze else out


def _oai(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """Clamped SwiGLU-OAI reference: gate clamped from above only, up
    clamped symmetrically, (up + 1) * g * sigmoid(1.702 * g)."""
    g = gate.float().clamp(max=limit)
    u = up.float().clamp(min=-limit, max=limit)
    return (u + 1.0) * g * torch.sigmoid(1.702 * g)


# ── Case A: routed MoE act-3 through the production entry point ─────────────


def _moe_layer(activation: str, **attrs):
    return SimpleNamespace(
        moe_config=SimpleNamespace(activation=SimpleNamespace(value=activation)),
        **attrs,
    )


def _case_a_layer(k: int, device: torch.device, *, real_inners: bool):
    """The fake routed layer exactly as the loader leaves it: gate params on
    the layer, inners per expert, fused pointer tables + temps."""
    packs = {
        p: _pack(
            A_HIDDEN if p != "down" else A_INTER,
            A_INTER if p != "down" else A_HIDDEN,
            k,
            device,
        )
        for p in ("gate", "up", "down")
    }
    if real_inners:
        inners = [
            {
                p: exl3.make_linear_exl3(
                    packs[p]["trellis"], packs[p]["suh"], packs[p]["svh"],
                    None, _mul1_marker(device),
                )
                for p in ("gate", "up", "down")
            }
            for _ in range(A_NEXP)
        ]
    else:
        class _Stub:
            def __init__(self, w):
                self.w = w

            def forward(self, h, _meta, out_dtype=None):
                return h @ self.w.t()

        inners = [
            {
                p: _Stub(
                    torch.randn(
                        (A_INTER if p != "down" else A_HIDDEN),
                        (A_HIDDEN if p != "down" else A_INTER),
                        dtype=torch.float16, device=device,
                    )
                    * 0.05
                )
                for p in ("gate", "up", "down")
            }
            for _ in range(A_NEXP)
        ]
    layer = _moe_layer(
        "swigluoai_uninterleave",
        swiglu_alpha=exl3.MINIMAX_SWIGLU_ALPHA_DEFAULT,
        swiglu_beta=1.0,
        swiglu_limit=exl3.MINIMAX_SWIGLU_LIMIT_DEFAULT,
        _exl3_inners=inners,
        _exl3_hidden_size=A_HIDDEN,
        _exl3_intermediate_local=A_INTER,
        _exl3_bits=k,
        _exl3_k=k,
        _exl3_mixed_k=False,
        # device discovery handle for build_exl3_fused_state (loader layout)
        w13_suh=torch.zeros(1, dtype=torch.float16, device=device),
    )
    if real_inners:
        # Loader-faithful codebook launch flags: the real inners carry the
        # mul1 marker (one per projection), so the loader's _flag_pack
        # (exl3.py:4588-4605) records (mcg, mul1) = (False, True) for every
        # projection. Without this the apply entry point would fall back to
        # the legacy MCG default and the case would test the wrong codebook.
        layer._exl3_codebook_flags = _loader_codebook_flags(
            [(False, True)] * 3
        )
    return layer, inners


def _case_a_probe(
    layer: Any,
    method: Any,
    raw: list[dict[str, Any]],
    k: int,
    tokens: int,
    device: torch.device,
) -> tuple[str, float, float]:
    """One apply at `tokens` rows: returns the dispatch label and the
    cosines against the act-3 and silu references. The dispatch label is
    stamped by the inner path itself (coop / fused / loop / native /
    mixed_k_reference); 'unknown' means no path stamped it."""
    x = torch.randn(tokens, A_HIDDEN, dtype=torch.float16, device=device) * 0.1
    weights = torch.softmax(
        torch.randn(tokens, A_NEXP, dtype=torch.float16, device=device), dim=-1
    )
    ids = weights.topk(TOPK, dim=-1).indices
    w_sel = weights.gather(1, ids)

    out = method.apply(layer, x, w_sel, ids, None, None)
    torch.cuda.synchronize()
    path = getattr(layer, "_exl3_last_apply", "unknown")

    limit = exl3.MINIMAX_SWIGLU_LIMIT_DEFAULT
    acc = torch.zeros(tokens, A_HIDDEN, dtype=torch.float32, device=device)
    acc_silu = torch.zeros_like(acc)
    xh = x.half()
    for e in range(A_NEXP):
        rw = ((ids == e).to(w_sel.dtype) * w_sel).sum(dim=1, keepdim=True)
        if rw.abs().max() == 0:
            continue
        g = _gemv_ref(xh, raw[e]["gate"], k).float()
        u = _gemv_ref(xh, raw[e]["up"], k).float()
        act = _oai(g, u, limit).half()
        act_silu = (torch.nn.functional.silu(g.clamp(max=limit))
                    * u.clamp(min=-limit, max=limit)).half()
        d = _gemv_ref(act, raw[e]["down"], k).float()
        d_silu = _gemv_ref(act_silu, raw[e]["down"], k).float()
        acc += rw.float() * d
        acc_silu += rw.float() * d_silu
    return path, _cos(out, acc), _cos(out, acc_silu)


def _case_a_run(k: int, device: torch.device) -> dict[str, Any]:
    layer, inners = _case_a_layer(k, device, real_inners=True)
    # The pack's codebook contract: every projection resolved to the mul1
    # codebook on BOTH sides (layer launch flags and the reference entry).
    assert layer._exl3_codebook_flags == (False, True) * 3
    for e in range(A_NEXP):
        for p in ("gate", "up", "down"):
            assert getattr(inners[e][p], "mcg", None) is False
            assert getattr(inners[e][p], "mul1", None) is True
    # Share the SAME synthetic packs with the reference: rebuild the raw
    # packs from the inners' own tensors so kernel and reference see
    # identical bytes.
    raw = [
        {
            p: {
                "trellis": inners[e][p].trellis,
                "suh": inners[e][p].suh,
                "svh": inners[e][p].svh,
            }
            for p in ("gate", "up", "down")
        }
        for e in range(A_NEXP)
    ]
    exl3.build_exl3_fused_state(layer, inners)
    cfg = exl3.Exl3Config.__new__(exl3.Exl3Config)
    method = exl3.Exl3MoEMethod(SimpleNamespace(), cfg, bits=k)

    # Decode shape: 64 tokens x topk 2 = 128 slots <= the coop kernel's
    # 256-slot cap -> the exl3_moe_coop act-3 fast path the serve takes at
    # decode (m=1 topk 4 = 4 slots, same kernel).
    path, cos, cos_silu = _case_a_probe(layer, method, raw, k, A_TOKENS, device)
    # Prefill shape: 192 tokens x topk 2 = 384 slots > the 256-slot cap ->
    # the expert-batched exl3_moe path the serve takes at prefill sizes
    # (real prefill chunk-slices into <=2048-row slices of the same path).
    prefill_path, prefill_cos, _ = _case_a_probe(
        layer, method, raw, k, A_PREFILL_TOKENS, device
    )

    return {
        "k": k,
        "path": path,
        "prefill_path": prefill_path,
        "codebook": "mul1",
        "cosine": round(cos, 6),
        "cosine_silu_reference": round(cos_silu, 6),
        "prefill_cosine": round(prefill_cos, 6),
        "pass": bool(
            cos >= COS_MIN
            and cos_silu < COS_MIN
            and path == "coop"
            and prefill_path == "batched"
            and prefill_cos >= COS_MIN
        ),
    }


# ── Case A2: the mixed-K dispatch the real pack's serve takes ───────────────


def _proj_k(pack: dict) -> int:
    """A projection's true EXL3 K, from the trellis width itself (16*K)."""
    return int(pack["trellis"].shape[-1]) // 16


def _case_a_mixedk_layer(device: torch.device):
    """Six synthetic experts in the real pack's mixed-K layout: two experts
    per distinct (K_gate, K_up, K_down) tuple, real LinearEXL3 inners with
    mul1 markers, and the loader's mixed-K fused state."""
    inners = []
    raw = []
    tuples = []
    for kt in A_MIXED_TUPLES:
        for _ in range(2):
            kg, ku, kd = kt
            packs = {
                "gate": _pack(A_HIDDEN, A_INTER, kg, device),
                "up": _pack(A_HIDDEN, A_INTER, ku, device),
                "down": _pack(A_INTER, A_HIDDEN, kd, device),
            }
            inner = {
                p: exl3.make_linear_exl3(
                    packs[p]["trellis"], packs[p]["suh"], packs[p]["svh"],
                    None, _mul1_marker(device),
                )
                for p in ("gate", "up", "down")
            }
            inners.append(inner)
            raw.append(packs)
            tuples.append(kt)
    # The synthetic packs must carry exactly the intended per-projection K
    # (the trellis width is 16*K), and the inners must carry the mul1 marker.
    for e, (kg, ku, kd) in enumerate(tuples):
        assert _proj_k(raw[e]["gate"]) == kg
        assert _proj_k(raw[e]["up"]) == ku
        assert _proj_k(raw[e]["down"]) == kd
        for p in ("gate", "up", "down"):
            assert getattr(inners[e][p], "mcg", None) is False
            assert getattr(inners[e][p], "mul1", None) is True
    layer = _moe_layer(
        "swigluoai_uninterleave",
        swiglu_alpha=exl3.MINIMAX_SWIGLU_ALPHA_DEFAULT,
        swiglu_beta=1.0,
        swiglu_limit=exl3.MINIMAX_SWIGLU_LIMIT_DEFAULT,
        _exl3_inners=inners,
        _exl3_hidden_size=A_HIDDEN,
        _exl3_intermediate_local=A_INTER,
        _exl3_mixed_k=True,
        w13_suh=torch.zeros(1, dtype=torch.float16, device=device),
    )
    layer._exl3_codebook_flags = _loader_codebook_flags(
        [(False, True)] * 3
    )
    exl3.build_exl3_fused_state(layer, inners)
    # The loader must have bucketed the experts into exactly the pack's
    # three K groups (two experts each) and disabled coop compatibility.
    assert [kt for kt, _ in layer._exl3_k_groups] == list(A_MIXED_TUPLES)
    assert [len(ids) for _, ids in layer._exl3_k_groups] == [2, 2, 2]
    assert layer._exl3_mixedk_unified is True
    assert layer._exl3_coop_compat is False
    return layer, raw, tuples


def _case_a_mixedk_probe(
    layer: Any,
    method: Any,
    raw: list[dict[str, Any]],
    tuples: list[tuple[int, int, int]],
    tokens: int,
    device: torch.device,
) -> tuple[str, float, float]:
    """One router-free apply at `tokens` rows (round-robin over the six
    experts so no expert reaches the fat threshold): the dispatch label and
    the cosines against the per-projection-K act-3 and silu references."""
    n_exp = len(raw)
    x = torch.randn(tokens, A_HIDDEN, dtype=torch.float16, device=device) * 0.1
    ids = (
        torch.arange(tokens * TOPK, device=device) % n_exp
    ).reshape(tokens, TOPK).long()
    w_sel = torch.ones(tokens, TOPK, device=device)

    out = method.apply(layer, x, w_sel, ids, None, None)
    torch.cuda.synchronize()
    path = getattr(layer, "_exl3_last_apply", "unknown")

    limit = exl3.MINIMAX_SWIGLU_LIMIT_DEFAULT
    acc = torch.zeros(tokens, A_HIDDEN, dtype=torch.float32, device=device)
    acc_silu = torch.zeros_like(acc)
    xh = x.half()
    for e in range(n_exp):
        rw = ((ids == e).to(w_sel.dtype) * w_sel).sum(dim=1, keepdim=True)
        if rw.abs().max() == 0:
            continue
        kg, ku, kd = tuples[e]
        g = _gemv_ref(xh, raw[e]["gate"], kg).float()
        u = _gemv_ref(xh, raw[e]["up"], ku).float()
        act = _oai(g, u, limit).half()
        act_silu = (torch.nn.functional.silu(g.clamp(max=limit))
                    * u.clamp(min=-limit, max=limit)).half()
        d = _gemv_ref(act, raw[e]["down"], kd).float()
        d_silu = _gemv_ref(act_silu, raw[e]["down"], kd).float()
        acc += rw.float() * d
        acc_silu += rw.float() * d_silu
    return path, _cos(out, acc), _cos(out, acc_silu)


def _case_a_mixedk_run(device: torch.device) -> dict[str, Any]:
    """The mixed-K dispatch table the serve runs for the real pack:
    default mode (unified exl3_moe_mixedk) and the explicit grouped mode,
    each at a decode-size and a prefill-size batch."""
    layer, raw, tuples = _case_a_mixedk_layer(device)
    cfg = exl3.Exl3Config.__new__(exl3.Exl3Config)
    method = exl3.Exl3MoEMethod(SimpleNamespace(), cfg, bits=3)
    probes = []
    try:
        for mode in ("default", "grouped"):
            if mode == "grouped":
                os.environ["VLLM_EXL3_MIXEDK_MODE"] = "grouped"
            else:
                os.environ.pop("VLLM_EXL3_MIXEDK_MODE", None)
            for tokens in A_MIXED_MS:
                path, cos, cos_silu = _case_a_mixedk_probe(
                    layer, method, raw, tuples, tokens, device
                )
                probes.append({
                    "mode": mode,
                    "tokens": tokens,
                    "path": path,
                    "cosine": round(cos, 6),
                    "cosine_silu_reference": round(cos_silu, 6),
                })
    finally:
        os.environ.pop("VLLM_EXL3_MIXEDK_MODE", None)
    expected = {
        ("default", 64): "mixedk_unified",
        ("default", 512): "mixedk_unified",
        ("grouped", 64): "mixedk_grouped",
        ("grouped", 512): "mixedk_grouped",
    }
    return {
        "tuples": [list(t) for t in A_MIXED_TUPLES],
        "probes": probes,
        "pass": all(
            p["path"] == expected[(p["mode"], p["tokens"])]
            and p["cosine"] >= COS_MIN
            and p["cosine_silu_reference"] < COS_MIN
            for p in probes
        ),
    }


# ── Case B: fused qkv + indexer linear ──────────────────────────────────────


def _e4m3_decode(payload: torch.Tensor) -> torch.Tensor:
    """Hand E4M3FN decode (independent of the plugin; mirrors torch's
    float8_e4m3fn semantics, verified in --cpu-selftest)."""
    bits = payload.to(torch.int32) & 0xFF
    sign = torch.where(bits & 0x80 != 0, -1.0, 1.0)
    exp = (bits >> 3) & 0xF
    man = bits & 0x7
    val = torch.where(
        exp == 0,
        man.to(torch.float32) * (2.0 ** -9),
        (man.to(torch.float32) + 8.0)
        * torch.pow(2.0, exp.to(torch.float32) - 10.0),
    )
    return sign * val


def _mxfp8_dequant(payload: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """F8_E4M3 payload x U8 E8M0 block-32 scales -> BF16 (hand decode; the
    scale byte is a direct exponent: multiply by 2**(byte - 127))."""
    k = payload.shape[-1]
    scale = torch.pow(
        2.0, scales.to(torch.float32) - 127.0
    ).repeat_interleave(32, dim=-1)[:, :k]
    return (_e4m3_decode(payload) * scale).to(torch.bfloat16)


def _finite_e4m3_bytes(shape, device):
    """Random F8_E4M3 payload bytes without the two NaN encodings
    (0x7F/0xFF), so kernel and reference comparisons stay finite."""
    b = torch.randint(0, 239, shape, dtype=torch.int32, device=device)
    b = b + (b >= 0x7F).to(torch.int32)  # {0..126, 128..240}
    return b.to(torch.uint8)


def _fused_layer(linears, indexer_payloads, indexer_scales, device):
    """The fused [q | k | v | index_q | index_k] layer exactly as the loader
    leaves it: per-shard LinearEXL3 objects for the EXL3 shards (None
    placeholders for the two MXFP8 indexer shards), the loader's bf16 shard
    list, and the BF16 staging rows built by the plugin's own
    dequant_minimax_indexer (the production staging step)."""
    layer = type(
        "MinimaxM3QKVParallelLinearWithIndexer", (SimpleNamespace,), {}
    )()
    layer._exl3_linears = list(linears) + [None, None]
    layer._exl3_linear_bf16_shards = [3, 4]
    layer._exl3_linear_output_partition_sizes = list(B_OUTS)
    layer._exl3_linear_padded = False
    layer._exl3_indexer_payloads = list(indexer_payloads)
    layer._exl3_indexer_scales = list(indexer_scales)
    layer._exl3_indexer_shards = (3, 4)
    staging = torch.zeros(
        sum(B_OUTS[3:]), B_IN, dtype=torch.bfloat16, device=device
    )
    slot = 0
    for i, (payload, scale) in enumerate(zip(indexer_payloads, indexer_scales)):
        w = exl3.dequant_minimax_indexer(payload, scale)
        staging[slot:slot + B_OUTS[3 + i]] = w.to(torch.bfloat16)
        slot += B_OUTS[3 + i]
    layer._exl3_bf16_weight = staging
    return layer


def _run_fused(layer, x, exl3_ref_fn, indexer_ref, ms):
    """Run the fused apply at each row count in ms and compare every output
    slice against per-slice references. ``exl3_ref_fn(i, xh)`` returns the
    reference for EXL3 slice i from the fp16 rows; ``indexer_ref[i]`` is
    the BF16 reference weight of indexer slice i."""
    cfg = exl3.Exl3Config.__new__(exl3.Exl3Config)
    method = exl3.Exl3LinearMethod(cfg, bits=B_K)
    names = ("q", "k", "v", "index_q", "index_k")
    per_m: dict[str, Any] = {}
    for m in ms:
        xm = x[:m].contiguous()
        out = method.apply(layer, xm)
        torch.cuda.synchronize()
        cos = {}
        for i, off in enumerate(B_OFFSETS):
            end = off + B_OUTS[i]
            sl = out[:, off:end]
            if i < 3:
                ref = exl3_ref_fn(i, xm.half()).float()
            else:
                ref = (xm.float() @ indexer_ref[i - 3].t().float()).float()
            cos[names[i]] = round(_cos(sl, ref), 6)
        per_m[f"m{m}"] = cos
    return per_m


def _case_b(device: torch.device) -> dict[str, Any]:
    # q/k/v synthetic EXL3 packs WITH the loader's mul1 marker: the real
    # pack ships a .mul1 scalar with every EXL3 tensor, the loader validates
    # it and hands it to LinearEXL3, and the dense paths decode the mul1
    # codebook. Markerless packs would decode the plain codebook on the
    # layer side and any mismatched reference would be uncorrelated.
    linears, ref_exl3 = [], []
    for out_f in B_OUTS[:3]:  # q, k, v
        pack = _pack(B_IN, out_f, B_K, device)
        linears.append(
            exl3.make_linear_exl3(
                pack["trellis"], pack["suh"], pack["svh"],
                None, _mul1_marker(device),
            )
        )
        ref_exl3.append(pack)
    for lin in linears:
        assert getattr(lin, "mcg", None) is False
        assert getattr(lin, "mul1", None) is True

    # The two MXFP8 indexer shards from synthetic F8_E4M3 + U8 E8M0 bytes.
    payloads, scales, ref_bf16 = [], [], []
    for out_f in B_OUTS[3:]:
        payload = _finite_e4m3_bytes((out_f, B_IN), device)
        scale = torch.randint(
            100, 130, (out_f, B_IN // 32), dtype=torch.int32, device=device
        ).to(torch.uint8)
        payloads.append(payload)
        scales.append(scale)
        ref_bf16.append(_mxfp8_dequant(payload, scale))
    layer = _fused_layer(linears, payloads, scales, device)

    def exl3_ref_fn(i, xh):
        # Reference decodes the SAME mul1 codebook (mcg=False maps to the
        # mul1 decoder in the native entry).
        return _gemv_ref(xh, ref_exl3[i], B_K, mcg=False)

    x = torch.randn(max(B_MS), B_IN, dtype=torch.bfloat16, device=device) * 0.1
    per_m = _run_fused(layer, x, exl3_ref_fn, ref_bf16, B_MS)
    return {
        "slices": per_m,
        "offsets": list(B_OFFSETS),
        "codebook": "mul1",
        "pass": all(
            v >= COS_MIN for cos in per_m.values() for v in cos.values()
        ),
    }


# ── Case C: rejection contract (no kernel launches) ────────────────────────


def _case_c(vllm_available: bool) -> dict[str, Any]:
    res: dict[str, Any] = {}

    def _raises(fn, exc):
        try:
            fn()
            return False
        except exc:
            return True
        except Exception as e:  # wrong exception type is a failure
            res.setdefault("wrong_exc", []).append(repr(e))
            return False

    def _alpha_2():
        exl3.resolve_moe_gate_params(
            _moe_layer(
                "swigluoai_uninterleave",
                swiglu_alpha=2.0, swiglu_beta=1.0, swiglu_limit=7.0,
            )
        )

    def _beta_05():
        exl3.resolve_moe_gate_params(
            _moe_layer(
                "swigluoai_uninterleave",
                swiglu_alpha=1.702, swiglu_beta=0.5, swiglu_limit=7.0,
            )
        )

    def _unknown_act():
        exl3.resolve_moe_gate_params(
            _moe_layer("gegelu", swiglu_alpha=1.702, swiglu_beta=1.0,
                       swiglu_limit=7.0)
        )

    res["alpha_2_raises"] = _raises(_alpha_2, ValueError)
    res["beta_05_raises"] = _raises(_beta_05, ValueError)
    res["unknown_activation_raises"] = _raises(_unknown_act, ValueError)

    res["tp_gt1_raises"] = None
    if vllm_available:
        # The TP>1 rejection lives in create_weights (it resolves per-layer
        # TP metadata before any allocation). It does NOT need a real vLLM
        # linear instance: the indexer-shard detection keys on the class
        # NAME (exl3.py _indexer_shards, type(layer).__name__) and the TP
        # geometry comes from the layer's declared tp_size
        # (exl3.py _resolve_tp_geometry reads it before vLLM's process-wide
        # state), so the probe is a plain class with the loader-relevant
        # metadata -- constructing QKVParallelLinear itself needs the full
        # vLLM distributed init and only produced a TypeError at __init__.
        import importlib

        lin = importlib.import_module("vllm.model_executor.layers.linear")
        assert hasattr(lin, "QKVParallelLinear")  # the dev image's vllm

        cfg = exl3.Exl3Config.__new__(exl3.Exl3Config)
        # from_config leaves non_routed_exl3 empty for this pack (the card
        # has no ledger), so _bf16_shards_for resolves [] on the real path.
        cfg.non_routed_exl3 = {}
        method = exl3.Exl3LinearMethod(cfg, bits=B_K)

        class MinimaxM3QKVParallelLinearWithIndexer:  # noqa: N801
            tp_size = 2
            prefix = "model.layers.0.self_attn.qkv_proj"

        layer = MinimaxM3QKVParallelLinearWithIndexer()
        # The rejection preconditions resolve without any vLLM layer.
        assert exl3.Exl3LinearMethod._indexer_shards(method, layer, 5) == (3, 4)
        assert exl3._resolve_tp_geometry(layer) == (0, 2)
        res["tp_gt1_raises"] = _raises(
            lambda: method.create_weights(
                layer,
                input_size_per_partition=B_IN,
                output_partition_sizes=list(B_OUTS),
                input_size=B_IN,
                output_size=sum(B_OUTS),
                params_dtype=torch.bfloat16,
            ),
            RuntimeError,
        )
    return res


# ── Case D: native-vLLM MXFP8 linear (DSpark draft format) on sm_121 ───────


# The selector's CUDA candidate list, pinned == vllm's own
# _POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA] (kernels/linear/__init__.py).
# The selection loop consults is_supported() then can_implement() in this
# order and returns the first that admits the device (init_mxfp8_linear_kernel);
# run 3 recorded the pick on the GB10 as FlashInferCutlassMxfp8LinearKernel
# with a NaN output. Case D now runs EVERY candidate that claims support on
# (12,1) so the receipt names the backend(s) that are actually correct.
D_CANDIDATES = (
    "FlashInferCutedslMxfp8LinearKernel",
    "FlashInferCutlassMxfp8LinearKernel",
    "MarlinMxfp8LinearKernel",
    "B12xMxfp8LinearKernel",
    "EmulationMxfp8LinearKernel",
    "HummingMxfp8LinearKernel",
    "FlashInferTrtllmMxfp8LinearKernel",
    "TorchMxFp8LinearKernel",
)

# Candidates whose apply() quantizes the ACTIVATION to MXFP8 inside the
# kernel (flashinfer.py mxfp8_e4m3_quantize / flashinfer_mxfp8_quantize_8x4,
# pytorch.py mxfp8_e4m3_quantize, b12x's internal requant, humming's
# kMxfp8Dynamic input schema): the reference must run the same
# quantize->dequant activation. Every other candidate consumes the raw
# activation -- Marlin is W8A16 (marlin.py apply_mxfp8_marlin_linear) and
# Emulation dequantizes the weight to BF16 at load (emulation.py:38-50,
# VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD default on) -- so their reference
# uses the raw BF16 activation against the dequantized weight.
D_ACT_QUANTIZING = frozenset({
    "FlashInferCutedslMxfp8LinearKernel",
    "FlashInferCutlassMxfp8LinearKernel",
    "FlashInferTrtllmMxfp8LinearKernel",
    "TorchMxFp8LinearKernel",
    "HummingMxfp8LinearKernel",
    "B12xMxfp8LinearKernel",
})


def _case_d_metrics(y: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    nan = int(torch.isnan(y).sum().item())
    return {
        "cosine": round(_cos(y, ref), 6) if nan == 0 else None,
        "max_abs": round(float(y.abs().max().item()), 6)
        if y.numel() and torch.isfinite(y).all()
        else None,
        "nan": nan,
    }


def _case_d(device: torch.device) -> dict[str, Any]:
    import vllm  # noqa: F401  (the dev image's own vLLM)

    from vllm.model_executor.kernels.linear import (
        Mxfp8LinearLayerConfig,
        _POSSIBLE_MXFP8_KERNELS,
        init_mxfp8_linear_kernel,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.platforms import PlatformEnum

    # The selector's own CUDA list is the source of truth; the pinned
    # D_CANDIDATES tuple must cover it exactly so an upstream reshuffle
    # cannot silently drop a candidate from the sweep.
    vllm_candidates = [
        k.__name__
        for k in _POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA]
    ]
    if sorted(vllm_candidates) != sorted(D_CANDIDATES):
        return {
            "error": (
                "vllm's CUDA MXFP8 candidate list changed: "
                f"{sorted(vllm_candidates)} != {sorted(D_CANDIDATES)}"
            ),
            "pass": False,
        }

    cfg = Mxfp8LinearLayerConfig(weight_shape=(D_N, D_K))

    # Random payload: the checkpoint format (block-32 E8M0 scales along K).
    # NOTE the staging dtype contract: the quantizer's payload IS
    # float8_e4m3fn and the kernel takes it as-is; converting the payload
    # to uint8 (a VALUE cast, not a bitcast) destroys the E4M3 bit
    # pattern (1.0 -> byte 1 -> 2**-9), which is what run-3's Case D did
    # -- the kernel was handed garbage bits (many of them the 0x7F/0xFF
    # NaN encodings) and the NaN was a TOOL artifact, not evidence about
    # sm_121. Bit-level access for the reference decode goes through
    # view(uint8), which is the bitcast the loader itself uses.
    w_bf16 = torch.randn(D_N, D_K, dtype=torch.float32, device=device) * 0.05
    x_bf16 = torch.randn(D_TOKENS, D_K, dtype=torch.float32, device=device) * 0.1
    w_payload, w_scale = mxfp8_e4m3_quantize(w_bf16, is_sf_swizzled_layout=False)
    x_payload, x_scale = mxfp8_e4m3_quantize(x_bf16, is_sf_swizzled_layout=False)
    assert w_payload.dtype == torch.float8_e4m3fn and w_payload.ndim == 2
    w_ref = _mxfp8_dequant(
        w_payload.view(torch.uint8), w_scale.view(D_N, -1)
    )
    x_ref = _mxfp8_dequant(
        x_payload.view(torch.uint8), x_scale.view(D_TOKENS, -1)
    )

    # Sanity payload: every weight byte decodes to 1.0 (E4M3 0x38) with an
    # all-127 E8M0 scale (descale 2**0), so the reference is exactly the row
    # sums of the activation. A backend that NaNs here fails on the most
    # degenerate legal input, which separates a real sm_121 kernel failure
    # from scale-swizzle/synthetic-data mishandling (a backend that passes
    # sanity but fails random has a layout bug, not a compute bug).
    one_byte = int(
        torch.ones((), dtype=torch.float32)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .item()
    )
    s_payload = torch.full((D_N, D_K), one_byte, dtype=torch.uint8, device=device)
    s_scale = torch.full((D_N, D_K // 32), 127, dtype=torch.uint8, device=device)
    s_ref = torch.ones(D_N, D_K, dtype=torch.bfloat16, device=device)

    x_raw = x_bf16.to(torch.bfloat16)

    def refs(backend: str) -> dict[str, torch.Tensor]:
        quantizing = backend in D_ACT_QUANTIZING
        return {
            "random": (x_ref @ w_ref.t()).float()
            if quantizing
            else (x_raw.float() @ w_ref.t().float()),
            "sanity": (x_ref @ s_ref.t()).float()
            if quantizing
            else (x_raw.float() @ s_ref.t().float()),
        }

    kernels = {
        k.__name__: k for k in _POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA]
    }
    results = []
    usable = []
    for name in D_CANDIDATES:
        cls = kernels[name]
        entry: dict[str, Any] = {"backend": name}
        sup, reason = cls.is_supported()
        if not sup:
            entry["supported"] = False
            entry["reason"] = (reason or "")[:200]
            results.append(entry)
            continue
        imp, reason = cls.can_implement(cfg)
        if not imp:
            entry["supported"] = False
            entry["reason"] = f"can_implement: {(reason or '')[:180]}"
            results.append(entry)
            continue
        entry["supported"] = True
        # The exact method path the draft's ModelOptMxFp8Config drives:
        # create_weights registers weight [N,K] E4M3 + weight_scale
        # [N,K/32] E8M0 (KMxfp8Static.create_weights), the loader fills
        # them, wkey.process asserts ndim/dtype, then
        # kernel.process_weights_after_loading, then apply ->
        # kernel.apply_weights. The staging below IS that state.
        reference = refs(name)
        for tag, payload, scale in (
            ("random", w_payload, w_scale.view(D_N, -1)),
            ("sanity", s_payload.view(torch.float8_e4m3fn), s_scale),
        ):
            layer = SimpleNamespace()
            layer.weight = torch.nn.Parameter(
                payload.contiguous(), requires_grad=False
            )
            layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
            layer.params_dtype = torch.bfloat16
            layer.input_size_per_partition = D_K
            layer.output_size_per_partition = D_N
            try:
                kernel = cls(cfg)
                kernel.process_weights_after_loading(layer)
                y = kernel.apply_weights(layer, x_raw)
                torch.cuda.synchronize()
                entry[tag] = _case_d_metrics(y, reference[tag])
            except Exception as exc:  # noqa: BLE001
                entry[tag] = {"error": str(exc)[:300]}
            finally:
                del layer
                torch.cuda.empty_cache()
        rand = entry.get("random") or {}
        if (
            rand.get("cosine") is not None
            and rand["cosine"] >= COS_MIN
            and rand.get("nan") == 0
        ):
            usable.append(name)
        results.append(entry)

    # What the serve picks today: the auto selector on this device.
    try:
        auto = type(init_mxfp8_linear_kernel(weight_shape=(D_N, D_K))).__name__
    except Exception as exc:  # noqa: BLE001
        auto = f"selection error: {str(exc)[:200]}"

    return {
        "auto_backend": auto,
        "candidates": results,
        "usable": usable,
        # The verdict is now "at least one candidate is correct on this
        # GB10 and is named in the receipt" -- the auto backend's NaN is a
        # finding the CHECKIN maps to the serve's backend-selection flag,
        # not a silent pass of the broken default.
        "pass": bool(usable),
    }


# ── Case E: real pack bytes, layer 0 fused qkv + indexer ────────────────────


E_FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "data",
    "minimax_m3_layer0_real_shards.json",
)


F_FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "data",
    "minimax_m3_layer3_real_experts.json",
)


def _fixture_entries(path: str = E_FIXTURE) -> dict[str, Any]:
    """The pinned census digest of the real-pack tensors a case reads
    (dtype, shape, bytes, shard file) -- pinned so a Hub-side change
    cannot pass silently."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _hf_range(token: str, start: int, length: int) -> bytes:
    """One authenticated HTTP range read. The token is sent ONLY to
    huggingface.co: the resolve URL answers 302 with a signed CDN URL, and
    the CDN request carries no Authorization. The token is never printed,
    logged, or placed in argv/env."""
    import urllib.error
    import urllib.request

    url = (
        f"https://huggingface.co/{E_REPO}/resolve/{E_REVISION}/{E_SHARD}"
    )
    rng = f"bytes={start}-{start + length - 1}"

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}", "Range": rng}
    )
    try:
        resp = opener.open(req, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code not in (301, 302, 303, 307, 308):
            raise
        cdn_url = exc.headers.get("Location")
        if not cdn_url:
            raise
        req = urllib.request.Request(cdn_url, headers={"Range": rng})
        resp = urllib.request.urlopen(req, timeout=600)
    with resp:
        if resp.status != 206:
            raise RuntimeError(
                f"range read of {E_SHARD} returned HTTP {resp.status}, "
                "refusing an unbounded read"
            )
        data = resp.read()
    if len(data) != length:
        raise RuntimeError(
            f"range read of {E_SHARD} returned {len(data)} of {length} bytes"
        )
    return data


def _shard_header(token: str) -> tuple[int, dict[str, Any]]:
    """Shard-1 safetensors header: 8-byte LE length + JSON with per-tensor
    data_offsets (relative to the end of the header)."""
    import struct

    raw = _hf_range(token, 0, 8)
    (hlen,) = struct.unpack("<Q", raw)
    if not 1024 < hlen < 64 * 2**20:
        raise RuntimeError(f"implausible shard header length {hlen}")
    header = json.loads(_hf_range(token, 8, hlen))
    return 8 + hlen, header


_TORCH_DTYPES = {
    # census dtype -> the checkpoint tensor's torch dtype. Tensors are
    # rebuilt from flat bytes through the plugin loader's
    # coerce_checkpoint_region (the reinterpret ats_alias._ats_view
    # performs), which refuses non-itemsize-aligned regions.
    "I16": torch.int16,
    "I32": torch.int32,
    "F16": torch.float16,
    "U8": torch.uint8,
    "F8_E4M3": torch.float8_e4m3fn,
}


def _exl3_definitional_ref(xh: torch.Tensor, linear) -> torch.Tensor:
    """Definitional dequant reference for a real EXL3 shard, valid for every
    K in 1..8 (no GEMV entry exists above K=4, so the K5 shards cannot be
    referenced through exl3_gemv): decode the transformed-basis weight with
    the shard's OWN mul1 codebook (ext.reconstruct, the decode the layer
    itself uses), then run the identical Hadamard/scale pipeline with an
    fp32 GEMM. Cross-family evidence comes from the m=1 arm, where the
    layer runs the small-m GEMV family against this reconstruct-built
    reference."""
    import exllamav3_ext as ext

    k = int(linear.trellis.shape[-1]) // 16
    w = torch.empty(
        (linear.in_features, linear.out_features),
        dtype=torch.half,
        device=xh.device,
    )
    ext.reconstruct(w, linear.trellis, k, False, True)
    xh2 = torch.empty_like(xh)
    ext.had_r_128(xh, xh2, linear.suh, None, 1.0)
    y0 = xh2.float() @ w.float()
    y = torch.empty_like(y0)
    ext.had_r_128(y0, y, None, linear.svh, 1.0)
    return y


def _case_e(device: torch.device, token_file: str | None) -> dict[str, Any]:
    entries = _fixture_entries()
    if not token_file:
        token_file = E_TOKEN_FILE
    if not os.path.isfile(token_file):
        return {
            "skip": (
                f"no HF token file at {token_file}; Case E (real pack "
                "bytes) needs --hf-token-file"
            )
        }
    with open(token_file, encoding="utf-8") as fh:
        token = fh.read().strip()
    if not token:
        return {"error": "empty HF token file", "pass": False}

    data_base, header = _shard_header(token)
    names = {
        "q_t": "language_model.model.layers.0.self_attn.q_proj.trellis",
        "q_suh": "language_model.model.layers.0.self_attn.q_proj.suh",
        "q_svh": "language_model.model.layers.0.self_attn.q_proj.svh",
        "q_mul1": "language_model.model.layers.0.self_attn.q_proj.mul1",
        "k_t": "language_model.model.layers.0.self_attn.k_proj.trellis",
        "k_suh": "language_model.model.layers.0.self_attn.k_proj.suh",
        "k_svh": "language_model.model.layers.0.self_attn.k_proj.svh",
        "k_mul1": "language_model.model.layers.0.self_attn.k_proj.mul1",
        "v_t": "language_model.model.layers.0.self_attn.v_proj.trellis",
        "v_suh": "language_model.model.layers.0.self_attn.v_proj.suh",
        "v_svh": "language_model.model.layers.0.self_attn.v_proj.svh",
        "v_mul1": "language_model.model.layers.0.self_attn.v_proj.mul1",
        "iq_w": "language_model.model.layers.0.self_attn.index_q_proj.weight",
        "iq_s": "language_model.model.layers.0.self_attn.index_q_proj.weight_scale_inv",
        "ik_w": "language_model.model.layers.0.self_attn.index_k_proj.weight",
        "ik_s": "language_model.model.layers.0.self_attn.index_k_proj.weight_scale_inv",
    }

    def fetch(tag: str) -> torch.Tensor:
        name = names[tag]
        ent = entries[name]
        hdr = header.get(name)
        if hdr is None:
            raise RuntimeError(f"{name} is not in the shard header")
        if hdr.get("dtype") != ent["dtype"]:
            raise RuntimeError(
                f"{name}: header dtype {hdr.get('dtype')} != census "
                f"{ent['dtype']}"
            )
        if list(hdr.get("shape", [])) != list(ent["shape"]):
            raise RuntimeError(
                f"{name}: header shape {hdr.get('shape')} != census "
                f"{ent['shape']}"
            )
        b, e = hdr["data_offsets"]
        if e - b != ent["bytes"]:
            raise RuntimeError(
                f"{name}: header span {e - b} != census {ent['bytes']} bytes"
            )
        raw = _hf_range(token, data_base + b, e - b)
        u8 = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
        # The loader's own region-coercion helper rebuilds the checkpoint
        # tensor from flat bytes (the reinterpret ats_alias._ats_view
        # performs); a hand-rolled flat reshape is what run-3 died on --
        # LinearEXL3 asserts the 3-D tile layout the checkpoint stores.
        t = exl3.coerce_checkpoint_region(
            u8, _TORCH_DTYPES[ent["dtype"]], ent["shape"]
        )
        if t.dtype == torch.float8_e4m3fn:
            # The F8_E4M3 payload arrives as float8 (the safetensors dtype)
            # and the loader views it to uint8 before the dequant, exactly
            # as its indexer staging does at load end.
            t = t.view(torch.uint8)
        return t

    # EXL3 shards with their real mul1 markers and mixed K (q K4, k/v K5).
    linears = []
    ks = []
    for p in ("q", "k", "v"):
        trellis = fetch(f"{p}_t")
        suh = fetch(f"{p}_suh")
        svh = fetch(f"{p}_svh")
        marker = fetch(f"{p}_mul1")
        if int(marker.reshape(-1)[0].item()) != exl3.MUL1_MARKER_SIGNED_INT32:
            raise RuntimeError(
                f"{p}_proj.mul1 marker is not the loader's mul1 int32"
            )
        # The loader's own trellis gate -- the exact geometry every loader
        # slot adopts -- runs BEFORE any LinearEXL3 construction, so a
        # mis-shaped fetch can never reach a kernel call.
        out_f = B_OUTS[("q", "k", "v").index(p)]
        k = exl3.assert_exl3_trellis_3d(trellis, B_IN, out_f)
        ks.append(k)
        linears.append(
            exl3.make_linear_exl3(trellis, suh, svh, None, marker.reshape(1))
        )
    if ks != [4, 5, 5]:
        raise RuntimeError(f"layer-0 q/k/v Ks are {ks}, expected [4, 5, 5]")

    # Indexer shards: real F8_E4M3 payloads + U8 E8M0 block-32 scales.
    iq_w, ik_w = fetch("iq_w"), fetch("ik_w")
    iq_s, ik_s = fetch("iq_s"), fetch("ik_s")
    layer = _fused_layer(linears, [iq_w, ik_w], [iq_s, ik_s], device)

    def exl3_ref_fn(i, xh):
        return _exl3_definitional_ref(xh, linears[i])

    indexer_ref = [
        _mxfp8_dequant(iq_w, iq_s),
        _mxfp8_dequant(ik_w, ik_s),
    ]
    x = torch.randn(max(E_MS), B_IN, dtype=torch.bfloat16, device=device) * 0.1
    per_m = _run_fused(layer, x, exl3_ref_fn, indexer_ref, E_MS)
    return {
        "slices": per_m,
        "ks": ks,
        "markers": "mul1",
        "offsets": list(B_OFFSETS),
        "pass": all(
            v >= COS_MIN for cos in per_m.values() for v in cos.values()
        ),
    }


# ── Case F: real pack routed experts, mixed-K batched dispatch ──────────────


def _case_f(device: torch.device, token_file: str | None) -> dict[str, Any]:
    """Three REAL layer-3 routed experts -- one per (K_gate, K_up, K_down)
    tuple of the pack's census -- evaluated router-free through the plugin's
    production MoE entry point at m in {1, 96}. This is the exact mixed-K
    batched dispatch the serve takes for this pack at EVERY batch shape
    (mixed-K layers never qualify for the coop kernel), against a
    definitional mul1-dequant + act-3 reference per projection."""
    entries = _fixture_entries(F_FIXTURE)
    if not token_file:
        token_file = E_TOKEN_FILE
    if not os.path.isfile(token_file):
        return {
            "skip": (
                f"no HF token file at {token_file}; Case F (real pack "
                "routed experts) needs --hf-token-file"
            )
        }
    with open(token_file, encoding="utf-8") as fh:
        token = fh.read().strip()
    if not token:
        return {"error": "empty HF token file", "pass": False}

    data_base, header = _shard_header(token)
    base = f"language_model.model.layers.{F_LAYER}.block_sparse_moe.experts"
    # w1 -> gate, w3 -> up, w2 -> down (the loader's own projection map,
    # ats_alias.py proj_of): (in_features, out_features) per checkpoint slot.
    proj_of = {"gate": ("w1", F_HIDDEN, F_INTER), "up": ("w3", F_HIDDEN, F_INTER),
               "down": ("w2", F_INTER, F_HIDDEN)}

    def fetch(name: str) -> torch.Tensor:
        ent = entries[name]
        hdr = header.get(name)
        if hdr is None:
            raise RuntimeError(f"{name} is not in the shard header")
        if hdr.get("dtype") != ent["dtype"]:
            raise RuntimeError(
                f"{name}: header dtype {hdr.get('dtype')} != census "
                f"{ent['dtype']}"
            )
        if list(hdr.get("shape", [])) != list(ent["shape"]):
            raise RuntimeError(
                f"{name}: header shape {hdr.get('shape')} != census "
                f"{ent['shape']}"
            )
        b, e = hdr["data_offsets"]
        if e - b != ent["bytes"]:
            raise RuntimeError(
                f"{name}: header span {e - b} != census {ent['bytes']} bytes"
            )
        raw = _hf_range(token, data_base + b, e - b)
        u8 = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
        return exl3.coerce_checkpoint_region(
            u8, _TORCH_DTYPES[ent["dtype"]], ent["shape"]
        )

    inners = []
    ks = []
    for e in F_EXPERTS:
        inner = {}
        tup = []
        for p, (w, in_f, out_f) in proj_of.items():
            trellis = fetch(f"{base}.{e}.{w}.trellis")
            suh = fetch(f"{base}.{e}.{w}.suh")
            svh = fetch(f"{base}.{e}.{w}.svh")
            marker = fetch(f"{base}.{e}.{w}.mul1")
            if int(marker.reshape(-1)[0].item()) != exl3.MUL1_MARKER_SIGNED_INT32:
                raise RuntimeError(
                    f"expert {e} {w}.mul1 marker is not the loader's mul1 int32"
                )
            # The loader's own geometry gate BEFORE any LinearEXL3
            # construction (run 3 died inside LinearEXL3 on a flat region).
            tup.append(exl3.assert_exl3_trellis_3d(trellis, in_f, out_f))
            inner[p] = exl3.make_linear_exl3(
                trellis, suh, svh, None, marker.reshape(1)
            )
        # (K_gate, K_up, K_down) == (w1, w3, w2) Ks, in proj_of order.
        ks.append((tup[0], tup[1], tup[2]))
        inners.append(inner)
    if ks != [(2, 2, 2), (2, 2, 3), (3, 3, 3)]:
        raise RuntimeError(f"layer-{F_LAYER} expert K tuples are {ks}")

    layer = _moe_layer(
        "swigluoai_uninterleave",
        swiglu_alpha=exl3.MINIMAX_SWIGLU_ALPHA_DEFAULT,
        swiglu_beta=1.0,
        swiglu_limit=exl3.MINIMAX_SWIGLU_LIMIT_DEFAULT,
        _exl3_inners=inners,
        _exl3_hidden_size=F_HIDDEN,
        _exl3_intermediate_local=F_INTER,
        _exl3_mixed_k=True,
        w13_suh=torch.zeros(1, dtype=torch.float16, device=device),
    )
    layer._exl3_codebook_flags = _loader_codebook_flags(
        [(False, True)] * 3
    )
    exl3.build_exl3_fused_state(layer, inners)
    assert [kt for kt, _ in layer._exl3_k_groups] == [
        (2, 2, 2), (2, 2, 3), (3, 3, 3)
    ]
    assert layer._exl3_coop_compat is False
    cfg = exl3.Exl3Config.__new__(exl3.Exl3Config)
    method = exl3.Exl3MoEMethod(SimpleNamespace(), cfg, bits=3)

    # Definitional reference per projection: the transformed-basis decode
    # with the shard's own mul1 codebook + the identical Hadamard/scale
    # pipeline + an fp32 GEMM (per-projection K comes from the trellis
    # width inside _exl3_definitional_ref).
    x = torch.randn(max(F_MS), F_HIDDEN, dtype=torch.float16, device=device) * 0.1
    refs = []
    for e, inner in enumerate(inners):
        g = _exl3_definitional_ref(x, inner["gate"])
        u = _exl3_definitional_ref(x, inner["up"])
        act = _oai(g, u, exl3.MINIMAX_SWIGLU_LIMIT_DEFAULT).half()
        d = _exl3_definitional_ref(act, inner["down"])
        refs.append(d)
        del g, u, act
    torch.cuda.empty_cache()

    per_m = {}
    for m in F_MS:
        slices = {}
        paths = []
        for e, eid in enumerate(F_EXPERTS):
            # Router-free DIRECT expert evaluation: every row of this apply
            # is forced onto one expert, so each slice is one expert's own
            # output against its own reference.
            ids = torch.full((m, 1), e, dtype=torch.long, device=device)
            w_sel = torch.ones(m, 1, device=device)
            out = method.apply(layer, x[:m], w_sel, ids, None, None)
            torch.cuda.synchronize()
            paths.append(getattr(layer, "_exl3_last_apply", "unknown"))
            slices[f"expert_{eid}"] = round(_cos(out, refs[e][:m]), 6)
        per_m[m] = {"paths": paths, "slices": slices}
    return {
        "ks": [list(t) for t in ks],
        "markers": "mul1",
        "per_m": {str(m): v for m, v in per_m.items()},
        "pass": all(
            set(v["paths"]) == {"mixedk_unified"}
            and all(c >= COS_MIN for c in v["slices"].values())
            for v in per_m.values()
        ),
    }


# ── CPU self-test ──────────────────────────────────────────────────────────


def _selftest() -> int:
    ok = True
    device = torch.device("cpu")

    def _check(name, cond):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        ok = ok and bool(cond)

    # E4M3 hand decode == torch's float8 cast (finite values).
    bits = torch.arange(256, dtype=torch.uint8)
    ref = bits.view(torch.float8_e4m3fn).to(torch.float32)
    got = _e4m3_decode(bits)
    fin = torch.isfinite(ref)
    _check("e4m3 hand decode == torch float8", bool(torch.allclose(got[fin], ref[fin])))

    # E8M0 semantics: dequant(x, s) == x_de * 2**(s - 127).
    payload = _finite_e4m3_bytes((4, 64), device)
    scales = torch.randint(100, 130, (4, 2), dtype=torch.int32).to(torch.uint8)
    got = _mxfp8_dequant(payload, scales)
    ref = (
        _e4m3_decode(payload)
        * torch.pow(2.0, scales.to(torch.float32) - 127.0).repeat_interleave(32, dim=-1)
    ).to(torch.bfloat16)
    _check("e8m0 block-32 descale", bool(torch.equal(got, ref)))

    # Case C rejections (pure python).
    try:
        import vllm  # noqa: F401

        vllm_available = True
    except Exception:
        vllm_available = False
    c = _case_c(vllm_available)
    _check("case C: alpha 2.0 raises", c["alpha_2_raises"])
    _check("case C: beta 0.5 raises", c["beta_05_raises"])
    _check("case C: unknown activation raises", c["unknown_activation_raises"])
    if c["tp_gt1_raises"] is not None:
        _check("case C: TP>1 indexer raises", c["tp_gt1_raises"])
    else:
        print("  skip case C: TP>1 (needs vllm; runs in the dev image)")

    # Entry point resolves act 3 + the MiniMax limit end-to-end (stub experts,
    # no kernel launch): the production apply() must refuse nothing and hand
    # the OAI activation to the loop, and the loop must match the reference.
    torch.manual_seed(0)
    layer, inners = _case_a_layer(2, device, real_inners=False)
    captured = {}

    def fake_apply(x, ids, weights, lay, limit=None, fused=None):
        captured["limit"] = limit
        captured["act"] = exl3.resolve_moe_act(lay)
        return exl3.apply_exl3_python_loop(
            x, ids, weights, lay._exl3_inners, None, limit,
            act=captured["act"],
        )

    orig = exl3.apply_exl3_experts
    exl3.apply_exl3_experts = fake_apply
    try:
        cfg = exl3.Exl3Config.__new__(exl3.Exl3Config)
        method = exl3.Exl3MoEMethod(SimpleNamespace(), cfg, bits=2)
        x = torch.randn(A_TOKENS, A_HIDDEN) * 0.1
        w = torch.softmax(torch.randn(A_TOKENS, A_NEXP), dim=-1)
        ids = w.topk(TOPK, dim=-1).indices
        w_sel = w.gather(1, ids)
        out = method.apply(layer, x, w_sel, ids, None, None)
    finally:
        exl3.apply_exl3_experts = orig
    _check("entry point resolves act 3", captured["act"] == exl3.MOE_ACT_SWIGLU_OAI)
    _check("entry point resolves the MiniMax limit", captured["limit"] == 7.0)
    # The loop output must match the hand OAI reference on the stub experts,
    # and must NOT match the silu reference computed on the same experts.
    acc = torch.zeros(A_TOKENS, A_HIDDEN)
    acc_silu = torch.zeros(A_TOKENS, A_HIDDEN)
    xh = x.half()
    for e in range(A_NEXP):
        rw = ((ids == e).to(w_sel.dtype) * w_sel).sum(dim=1, keepdim=True)
        g = (xh @ inners[e]["gate"].w.t()).clamp(max=7.0)
        u = (xh @ inners[e]["up"].w.t()).clamp(min=-7.0, max=7.0)
        act = ((u + 1.0) * g * torch.sigmoid(1.702 * g)).half()
        act_silu = (torch.nn.functional.silu(g) * u).half()
        acc += rw * (act @ inners[e]["down"].w.t()).float()
        acc_silu += rw * (act_silu @ inners[e]["down"].w.t()).float()
    _check("loop output == hand OAI reference", _cos(out, acc) >= COS_MIN)
    _check("act 3 diverges from the silu reference",
           _cos(out, acc_silu) < COS_MIN)

    # Case B pack/reference construction on CPU: the plugin's indexer dequant
    # must equal the hand decode, and the fused slice ORDER must be
    # [q | k | v | index_q | index_k].
    payload = _finite_e4m3_bytes((B_OUTS[3], B_IN), device)
    scale = torch.randint(100, 130, (B_OUTS[3], B_IN // 32), dtype=torch.int32).to(torch.uint8)
    got = exl3.dequant_minimax_indexer(payload, scale)
    ref = _mxfp8_dequant(payload, scale)
    _check("indexer dequant == hand reference", bool(torch.allclose(got.float(), ref.float())))
    offs = list(B_OFFSETS) + [sum(B_OUTS)]
    deltas = [offs[i + 1] - offs[i] for i in range(5)]
    _check("fused slice order/offsets", deltas == list(B_OUTS))

    # Codebook contract (the GPU_JOB-2 run-1 Case B failure): the pack's
    # EXL3 tensors carry the mul1 marker, so BOTH sides of every comparison
    # must resolve the mul1 codebook. The reference GEMV entry must default
    # to mcg=False (mcg=False maps to the mul1 decoder) -- a True default
    # decodes a different codebook and is uncorrelated by construction.
    import inspect

    sig = inspect.signature(_gemv_ref)
    _check("gemv reference defaults to the mul1 codebook",
           sig.parameters["mcg"].default is False)
    _check("loader marker value is the mul1 int32",
           int(_mul1_marker(device)[0].item()) == exl3.MUL1_MARKER_SIGNED_INT32)
    flags = _loader_codebook_flags([(False, True)] * 3)
    _check("loader codebook flags are (mcg, mul1) per projection",
           flags == (False, True) * 3)
    try:
        _loader_codebook_flags([(False, True)] * 2)
        _check("loader codebook flags reject wrong arity", False)
    except ValueError:
        _check("loader codebook flags reject wrong arity", True)

    # Case C preconditions resolve WITHOUT any vLLM layer: the indexer
    # shard detection keys on the class name and the TP geometry on the
    # declared tp_size, so the TP>1 rejection is exercisable through a
    # plain probe class (constructing QKVParallelLinear itself only
    # produced a TypeError at __init__ in run 1).
    class MinimaxM3QKVParallelLinearWithIndexer:  # noqa: N801
        tp_size = 2

    probe = MinimaxM3QKVParallelLinearWithIndexer()
    method = exl3.Exl3LinearMethod(
        exl3.Exl3Config.__new__(exl3.Exl3Config), bits=B_K
    )
    _check("case C probe resolves the indexer shards",
           method._indexer_shards(probe, len(B_OUTS)) == (3, 4))
    _check("case C probe resolves tp_size 2",
           exl3._resolve_tp_geometry(probe) == (0, 2))

    # Case E fixture consistency (offline): the pinned layer-0 digest must
    # describe exactly the fused layer's tensors with the pack's mixed K.
    entries = _fixture_entries()
    ok_e = len(entries) == 16 and all(
        v.get("file") == E_SHARD for v in entries.values()
    )
    ks = []
    for p in ("q", "k", "v"):
        t = entries[f"language_model.model.layers.0.self_attn.{p}_proj.trellis"]
        suh = entries[f"language_model.model.layers.0.self_attn.{p}_proj.suh"]
        svh = entries[f"language_model.model.layers.0.self_attn.{p}_proj.svh"]
        mul1 = entries[f"language_model.model.layers.0.self_attn.{p}_proj.mul1"]
        in_f = t["shape"][0] * 16
        ok_e = ok_e and t["bytes"] == 2 * t["shape"][0] * t["shape"][1] * t["shape"][2]
        ok_e = ok_e and suh["shape"] == [in_f] and svh["shape"] == [B_OUTS[("q", "k", "v").index(p)]]
        ok_e = ok_e and mul1["shape"] == [] and mul1["dtype"] == "I32"
        ks.append(t["shape"][2] // 16)
    ok_e = ok_e and ks == [4, 5, 5]
    for p in ("index_q", "index_k"):
        w = entries[f"language_model.model.layers.0.self_attn.{p}_proj.weight"]
        s = entries[f"language_model.model.layers.0.self_attn.{p}_proj.weight_scale_inv"]
        ok_e = ok_e and w["shape"][1] == B_IN and s["shape"] == [w["shape"][0], B_IN // 32]
    _check("case E fixture is the 16-tensor layer-0 digest (K q4/k5/v5)",
           bool(ok_e))
    _check("case E repo/revision pinned",
           E_REPO == "vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw"
           and E_REVISION == "687cbff2c210b7514d18c5f6a4d1db3417b316c4")

    # Case E tensors from the FIXTURE SHAPES, through the loader's own
    # helpers, BEFORE any kernel call: the fetched region must become the
    # 3-D [in/16, out/16, 16*K] trellis the loader adopts (run-3 died
    # inside LinearEXL3 because the tool kept the region flat), and the
    # loader's gate must reject the flat region and every wrong geometry.
    ok_e3d = True
    for p in ("q", "k", "v"):
        ent = entries[f"language_model.model.layers.0.self_attn.{p}_proj.trellis"]
        flat = torch.zeros(ent["bytes"], dtype=torch.uint8)
        t = exl3.coerce_checkpoint_region(flat, torch.int16, ent["shape"])
        out_f = B_OUTS[("q", "k", "v").index(p)]
        ok_e3d = ok_e3d and exl3.assert_exl3_trellis_3d(t, B_IN, out_f) == (
            ent["shape"][2] // 16
        )
        try:  # the flat region itself must be rejected by the gate
            exl3.assert_exl3_trellis_3d(flat.view(torch.int16), B_IN, out_f)
            ok_e3d = False
        except ValueError:
            pass
    _check("case E trellis reshapes to the loader's 3-D layout", ok_e3d)

    # Case D staging contract (the run-3 NaN): a `.to(torch.uint8)`
    # roundtrip of an E4M3 payload is a VALUE cast that destroys the bit
    # pattern (1.0 -> byte 1), while view(torch.uint8) is the bitcast. The
    # run-3 tool staged weights through the value cast, so every weight
    # collapsed to byte 0 and the kernel output was the zero vector --
    # cosine(0, ref) = nan. The fixed tool must stage the payload as-is.
    val_cast = int(
        torch.ones((), dtype=torch.float32)
        .to(torch.float8_e4m3fn).to(torch.uint8).item()
    )
    bit_cast = int(
        torch.ones((), dtype=torch.float32)
        .to(torch.float8_e4m3fn).view(torch.uint8).item()
    )
    _check("e4m3 value cast destroys the bit pattern (1 vs 56)",
           val_cast == 1 and bit_cast == 56)
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "gb10_minimax_m3_parity.py"), encoding="utf-8").read()
    # Case D's own function body, so this check's text cannot satisfy it.
    seg = src.split("def _case_d(", 1)[1].split("# ── Case E", 1)[0]
    _check("case D stages the fp8 payload as-is (no value cast)",
           ".to(torch.uint8)" not in seg
           and "s_payload.view(torch.float8_e4m3fn)" in seg
           and "w_payload.view(torch.uint8)" in seg)

    # Case D sweep coverage: the pinned candidate list is exactly vllm's
    # CUDA list with no duplicates, and the per-family activation
    # classification matches each kernel's apply (Marlin is W8A16,
    # Emulation dequantizes to BF16 at load; everything else quantizes
    # the activation in-kernel).
    _check("case D candidate list is the pinned 8 with no duplicates",
           sorted(D_CANDIDATES) == sorted(set(D_CANDIDATES)) and len(D_CANDIDATES) == 8
           and "EmulationMxfp8LinearKernel" in D_CANDIDATES
           and "MarlinMxfp8LinearKernel" in D_CANDIDATES
           and "TorchMxFp8LinearKernel" in D_CANDIDATES)
    _check("case D activation classification excludes marlin/emulation",
           D_ACT_QUANTIZING == frozenset(D_CANDIDATES) - {
               "MarlinMxfp8LinearKernel", "EmulationMxfp8LinearKernel",
           })
    # The sanity payload must decode to exactly 1.0 everywhere (E4M3 0x38
    # under an all-127 E8M0 scale), so the sanity reference is the row
    # sums of the activation.
    s_payload = torch.full((2, 64), 56, dtype=torch.uint8)
    s_scale = torch.full((2, 2), 127, dtype=torch.uint8)
    _check("case D sanity payload decodes to all ones",
           bool(((_mxfp8_dequant(s_payload, s_scale) == 1.0).all())))

    # Case A dispatch observability: the coop fast path must stamp its own
    # label (run 3 printed the generic "fused" entry label because the
    # coop branch returned early and the wrapper clobbered it, so the
    # path bar was unsatisfiable and the receipt said nothing about which
    # inner kernel ran). The wrapper must stamp "fused" only when the
    # inner call left the label unset, and the tool must read the label
    # with an "unknown" default and require BOTH the coop decode path and
    # a correct prefill-family probe.
    esrc = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "src", "vllm_exl3", "exl3.py"),
                encoding="utf-8").read()
    coop_seg = esrc.split("exl3_moe_coop(", 1)[1]
    _check("coop fast path stamps its own dispatch label",
           'layer._exl3_last_apply = "coop"' in coop_seg.split("return out", 1)[0])
    fused_idx = esrc.find('layer._exl3_last_apply = "fused"')
    _check("wrapper no longer clobbers the inner dispatch label",
           fused_idx > 0 and "is None:" in esrc[fused_idx - 200:fused_idx])
    _check("case A bar requires the coop path AND a prefill probe",
           'path == "coop"' in src and "prefill_cos >= COS_MIN" in src)
    _check("case A prefill probe is past the coop slot cap",
           A_PREFILL_TOKENS * TOPK > 256 and A_TOKENS * TOPK <= 256)
    _check("case A reads the label with an unknown default",
           'getattr(layer, "_exl3_last_apply", "unknown")' in src)

    # Run 4's blocker: the prefill (>256 slots) expert-batched exl3_moe path
    # scored cosine ~0.10 while the coop decode path matched at 1.0. The
    # python launch was proven faithful (tests/test_moe_prefill_dispatch_
    # contract.py mock-records both launches and compares act index,
    # codebook flags, K and pointer tables); the divergence was the fork
    # kernel's activation switch, fixed on branch feat/moe-act3-batched.
    # The tool must now (a) name the batched path in the receipt and bar
    # on its true label, (b) cover the mixed-K dispatch the real pack's
    # serve takes (uniform coop is never qualified for the real pack), and
    # (c) cover the real pack's own experts (Case F).
    _check("case A bar requires the batched prefill path by name",
           'prefill_path == "batched"' in src)
    _check("every batched inner path stamps its own label",
           'layer._exl3_last_apply = "batched"' in esrc
           and 'layer._exl3_last_apply = "mixedk_grouped"' in esrc
           and 'layer._exl3_last_apply = "mixedk_unified"' in esrc)
    _check("case A2 probes the pack's three K tuples at both modes",
           list(A_MIXED_TUPLES) == [(2, 2, 2), (2, 2, 3), (3, 3, 3)]
           and A_MIXED_MS == (64, 512))
    a2_seg = src.split("def _case_a_mixedk_run", 1)[1].split("# ── Case B", 1)[0]
    _check("case A2 bars each probe on its pinned true path label",
           '"mixedk_unified"' in a2_seg
           and '"mixedk_grouped"' in a2_seg
           and 'p["path"] == expected[(p["mode"], p["tokens"])]' in a2_seg)

    # Case F fixture consistency (offline): the pinned layer-3 digest covers
    # exactly one expert per distinct K tuple, all with mul1 markers, all in
    # shard 1, and the per-projection Ks derived from the trellis widths are
    # the pack's census tuples.
    fentries = _fixture_entries(F_FIXTURE)
    ok_f = len(fentries) == 36 and all(
        v.get("file") == E_SHARD for v in fentries.values()
    )
    fks = []
    for e in F_EXPERTS:
        tup = []
        for w in ("w1", "w3", "w2"):
            name = (
                f"language_model.model.layers.{F_LAYER}.block_sparse_moe"
                f".experts.{e}.{w}"
            )
            t = fentries[name + ".trellis"]
            ok_f = ok_f and t["dtype"] == "I16"
            ok_f = ok_f and fentries[name + ".mul1"]["shape"] == []
            ok_f = ok_f and fentries[name + ".mul1"]["dtype"] == "I32"
            ok_f = ok_f and (
                t["bytes"] == 2 * t["shape"][0] * t["shape"][1] * t["shape"][2]
            )
            tup.append(t["shape"][2] // 16)
        fks.append(tuple(tup))
    ok_f = ok_f and fks == [(2, 2, 2), (2, 2, 3), (3, 3, 3)]
    _check("case F fixture is the 3-expert per-K-tuple digest", bool(ok_f))
    f_seg = src.split("def _case_f", 1)[1]
    _check("case F bars the mixedk unified path at both m",
           'set(v["paths"]) == {"mixedk_unified"}' in f_seg
           and F_MS == (1, 96))

    print("GB10_MINIMAX_M3_PARITY SELFTEST " + ("PASS" if ok else "FAIL")
          + " (no hardware parity claim)")
    return 0 if ok else 1


# ── main ───────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="receipt path")
    ap.add_argument("--hf-token-file", default=None,
                    help="path to the Hub token file for Case E (real pack "
                    "bytes); the token is never printed or placed in "
                    "argv/env")
    ap.add_argument("--cpu-selftest", action="store_true",
                    help="CPU only: pack/reference construction + rejections; "
                    "no hardware parity claim")
    args = ap.parse_args()

    if args.cpu_selftest:
        return _selftest()

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device (run on the GB10 inside the dev image)")
        return 2
    device = torch.device("cuda", torch.cuda.current_device())
    receipt: dict[str, Any] = {
        "capability": list(torch.cuda.get_device_capability(device)),
        "device": torch.cuda.get_device_name(device),
    }
    print(json.dumps(receipt))

    verdict = True

    receipt["case_a"] = [_case_a_run(k, device) for k in (2, 3)]
    for ent in receipt["case_a"]:
        print(f"case A K={ent['k']}: path={ent['path']} cosine={ent['cosine']} "
              f"cosine_silu={ent['cosine_silu_reference']} "
              f"prefill_path={ent['prefill_path']} "
              f"prefill_cosine={ent['prefill_cosine']}")
        verdict = verdict and ent["pass"]

    receipt["case_a_mixedk"] = _case_a_mixedk_run(device)
    for ent in receipt["case_a_mixedk"]["probes"]:
        print(f"case A2 mode={ent['mode']} tokens={ent['tokens']}: "
              f"path={ent['path']} cosine={ent['cosine']} "
              f"cosine_silu={ent['cosine_silu_reference']}")
    verdict = verdict and receipt["case_a_mixedk"]["pass"]

    receipt["case_b"] = _case_b(device)
    print(f"case B slices: {receipt['case_b']['slices']}")
    verdict = verdict and receipt["case_b"]["pass"]

    try:
        import vllm  # noqa: F401

        vllm_available = True
    except Exception:
        vllm_available = False
    receipt["case_c"] = _case_c(vllm_available)
    print(f"case C: {receipt['case_c']}")
    verdict = verdict and all(
        v for v in receipt["case_c"].values() if v is not None
    )

    receipt["case_d"] = _case_d(device)
    d = receipt["case_d"]
    print(f"case D auto_backend={d.get('auto_backend')} usable={d.get('usable')}")
    for ent in d.get("candidates", []):
        print(f"  case D candidate: {ent}")
    verdict = verdict and d["pass"]

    receipt["case_e"] = _case_e(device, args.hf_token_file)
    if "skip" in receipt["case_e"]:
        print(f"case E: SKIP ({receipt['case_e']['skip']})")
    else:
        print(f"case E: {receipt['case_e']}")
        verdict = verdict and bool(receipt["case_e"].get("pass"))

    receipt["case_f"] = _case_f(device, args.hf_token_file)
    if "skip" in receipt["case_f"]:
        print(f"case F: SKIP ({receipt['case_f']['skip']})")
    else:
        print(f"case F: {receipt['case_f']}")
        verdict = verdict and bool(receipt["case_f"].get("pass"))

    receipt["mem_peak_gib"] = round(
        torch.cuda.max_memory_allocated() / 2**30, 3
    )
    print(f"peak torch.cuda.max_memory_allocated: {receipt['mem_peak_gib']} GiB")
    if receipt["mem_peak_gib"] >= MEM_BUDGET_GIB:
        verdict = False

    receipt["verdict"] = "PASS" if verdict else "FAIL"
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(receipt, fh, indent=2)
    print(f"GB10_MINIMAX_M3_PARITY {receipt['verdict']}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
