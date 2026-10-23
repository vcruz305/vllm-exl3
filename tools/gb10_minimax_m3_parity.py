#!/usr/bin/env python3
"""GB10 MiniMax-M3 EXL3 plugin parity probe (GPU_JOB-2).

Verifies, on the DGX Spark (GB10, sm_121), inside the dev image with this
worktree bind-mounted at /plugin and PYTHONPATH=/plugin/src:

  Case A  the production MoE entry point -- Exl3MoEMethod.apply on a layer
          built exactly as the loader leaves it -- resolves the MiniMax-M3
          act-3 (swiglu_oai) COOP kernel and matches a per-expert
          exl3_gemv reference with clamped SwiGLU-OAI, at K in {2, 3};
  Case B  the fused [q | k | v | index_q | index_k] linear: q/k/v EXL3
          trellis GEMMs plus the two MXFP8 indexer shards dequantized at
          load into BF16 side rows, sliced at the branch offsets, matching
          a per-shard reference in the branch's order;
  Case C  the rejection contract (CPU-side): wrong alpha/beta/activation
          raise, TP>1 indexer shards raise; no kernel launches;
  Case D  native-vLLM MXFP8 linear (the DSpark draft's format) at
          6144->12288 with block-32 E8M0 scales vs a dequant reference,
          recording the backend the sm_121 selector picks.

``--cpu-selftest`` runs everything that needs no kernel launch (Case C in
full, the pack/reference construction of A/B/D, the entry-point act
resolution through the real apply() with matmul-stub experts, and the
hand E4M3/E8M0 decode) on any box; the GPU arms run only with CUDA.

    python3 tools/gb10_minimax_m3_parity.py --cpu-selftest
    python3 tools/gb10_minimax_m3_parity.py --json receipt.json
"""

from __future__ import annotations

import argparse
import json
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

# Case B shapes: the branch's fused layer at TP1 (hidden 6144; q 8192,
# k 512, v 512, index_q 512, index_k 128). Synthetic packs ~30 MB total.
B_IN = 6144
B_OUTS = (8192, 512, 512, 512, 128)
B_OFFSETS = (0, 8192, 8704, 9216, 9728)  # vllm linear.py fused split order
B_K = 4  # synthetic EXL3 K for q/k/v (the pack's q/k/v are K4/K5)

# Case D shapes: the DSpark draft's dense FFN projection (hidden 6144 ->
# dspark_ffn_hidden_size 12288), MXFP8 block [1, 32] E8M0 scales.
D_N, D_K, D_TOKENS = 12288, 6144, 64


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


def _gemv_ref(x: torch.Tensor, pack: dict, k: int) -> torch.Tensor:
    """Per-expert dequant GEMM reference (the proven native-map entry)."""
    import vllm_exl3_c

    return vllm_exl3_c.exl3_gemv(
        x, pack["trellis"], pack["suh"], pack["svh"], k, True
    )


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
                    packs[p]["trellis"], packs[p]["suh"], packs[p]["svh"]
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
    return layer, inners


def _case_a_run(k: int, device: torch.device) -> dict[str, Any]:
    layer, inners = _case_a_layer(k, device, real_inners=True)
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

    x = torch.randn(A_TOKENS, A_HIDDEN, dtype=torch.float16, device=device) * 0.1
    weights = torch.softmax(
        torch.randn(A_TOKENS, A_NEXP, dtype=torch.float16, device=device), dim=-1
    )
    ids = weights.topk(TOPK, dim=-1).indices
    w_sel = weights.gather(1, ids)

    out = method.apply(layer, x, w_sel, ids, None, None)
    torch.cuda.synchronize()
    path = getattr(layer, "_exl3_last_apply", "coop")

    limit = exl3.MINIMAX_SWIGLU_LIMIT_DEFAULT
    acc = torch.zeros(A_TOKENS, A_HIDDEN, dtype=torch.float32, device=device)
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
    cos = _cos(out, acc)
    cos_silu = _cos(out, acc_silu)
    return {
        "k": k,
        "path": path,
        "cosine": round(cos, 6),
        "cosine_silu_reference": round(cos_silu, 6),
        "pass": bool(cos >= COS_MIN and cos_silu < COS_MIN and path == "coop"),
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


def _case_b(device: torch.device) -> dict[str, Any]:
    class MinimaxM3QKVParallelLinearWithIndexer:  # noqa: N801
        pass

    layer = MinimaxM3QKVParallelLinearWithIndexer()
    linears = []
    ref_exl3 = []
    for out_f in B_OUTS[:3]:  # q, k, v
        pack = _pack(B_IN, out_f, B_K, device)
        linears.append(
            exl3.make_linear_exl3(pack["trellis"], pack["suh"], pack["svh"])
        )
        ref_exl3.append(pack)
    linears += [None, None]
    layer._exl3_linears = linears
    layer._exl3_linear_bf16_shards = [3, 4]
    layer._exl3_linear_output_partition_sizes = list(B_OUTS)
    layer._exl3_linear_padded = False

    # The two MXFP8 indexer shards, through the plugin's load-time dequant
    # (the production staging step), from synthetic F8_E4M3 + U8 E8M0 bytes.
    payloads, scales, ref_bf16 = [], [], []
    for out_f in B_OUTS[3:]:
        payload = _finite_e4m3_bytes((out_f, B_IN), device)
        scale = torch.randint(
            100, 130, (out_f, B_IN // 32), dtype=torch.int32, device=device
        ).to(torch.uint8)
        payloads.append(payload)
        scales.append(scale)
        ref_bf16.append(_mxfp8_dequant(payload, scale))
    layer._exl3_indexer_payloads = payloads
    layer._exl3_indexer_scales = scales
    layer._exl3_indexer_shards = (3, 4)
    staging = torch.zeros(
        sum(B_OUTS[3:]), B_IN, dtype=torch.bfloat16, device=device
    )
    slot = 0
    for i, (payload, scale) in enumerate(zip(payloads, scales)):
        w = exl3.dequant_minimax_indexer(payload, scale)
        staging[slot:slot + B_OUTS[3 + i]] = w.to(torch.bfloat16)
        slot += B_OUTS[3 + i]
    layer._exl3_bf16_weight = staging

    cfg = exl3.Exl3Config.__new__(exl3.Exl3Config)
    method = exl3.Exl3LinearMethod(cfg, bits=B_K)
    x = torch.randn(64, B_IN, dtype=torch.bfloat16, device=device) * 0.1
    out = method.apply(layer, x)
    torch.cuda.synchronize()

    slices, refs, names = [], [], []
    xh = x.half()
    for i, off in enumerate(B_OFFSETS):
        end = off + B_OUTS[i]
        slices.append(out[:, off:end])
        if i < 3:
            refs.append(
                _gemv_ref(xh, ref_exl3[i], B_K).float().unsqueeze(0)
            )
        else:
            refs.append(
                (
                    x @ ref_bf16[i - 3].t()
                ).float()
            )
        names.append(("q", "k", "v", "index_q", "index_k")[i])
    cos = {
        names[i]: round(_cos(slices[i], refs[i]), 6) for i in range(5)
    }
    return {
        "slices": cos,
        "offsets": list(B_OFFSETS),
        "pass": all(v >= COS_MIN for v in cos.values()),
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
        # TP metadata before any allocation); exercise it with the real vllm
        # linear class and a declared tp_size of 2.
        import importlib

        lin = importlib.import_module("vllm.model_executor.layers.linear")

        class MinimaxM3QKVParallelLinearWithIndexer(  # noqa: N801
            lin.QKVParallelLinear
        ):
            tp_size = 2

        method = exl3.Exl3LinearMethod(
            exl3.Exl3Config.__new__(exl3.Exl3Config), bits=B_K
        )
        layer = MinimaxM3QKVParallelLinearWithIndexer()
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


def _case_d(device: torch.device) -> dict[str, Any]:
    import vllm  # noqa: F401  (the dev image's own vLLM)

    from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    w_bf16 = (torch.randn(D_N, D_K, dtype=torch.float32, device=device) * 0.05)
    x_bf16 = (torch.randn(D_TOKENS, D_K, dtype=torch.float32, device=device) * 0.1)

    # MXFP8 quantize (block-32 E8M0 scales along K), the checkpoint format.
    w_payload, w_scale = mxfp8_e4m3_quantize(w_bf16, is_sf_swizzled_layout=False)
    x_payload, x_scale = mxfp8_e4m3_quantize(x_bf16, is_sf_swizzled_layout=False)

    # Reference: hand-decode the same payload/scale bytes to BF16 and matmul
    # (the kernel quantizes the activation itself, so the reference uses the
    # same quantize-dequant activation; the compared quantity is the GEMM).
    w_ref = _mxfp8_dequant(w_payload.view(-1, D_K), w_scale.view(D_N, -1))
    x_ref = _mxfp8_dequant(x_payload.view(-1, D_K), x_scale.view(D_TOKENS, -1))
    y_ref = (x_ref @ w_ref.t()).float()

    try:
        kernel = init_mxfp8_linear_kernel(weight_shape=(D_N, D_K))
        backend = type(kernel).__name__
    except Exception as exc:  # noqa: BLE001
        return {
            "backend": None,
            "selection_error": str(exc)[:500],
            "pass": False,
        }

    layer = SimpleNamespace()
    layer.weight = torch.nn.Parameter(
        w_payload.view(-1, D_K).to(torch.uint8).view(torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        w_scale.view(D_N, -1).to(torch.uint8), requires_grad=False
    )
    layer.params_dtype = torch.bfloat16
    kernel.process_weights_after_loading(layer)
    try:
        y = kernel.apply_weights(layer, x_bf16.to(torch.bfloat16))
        torch.cuda.synchronize()
        cos = _cos(y, y_ref)
        return {
            "backend": backend,
            "cosine": round(cos, 6),
            "pass": bool(cos >= COS_MIN),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "backend": backend,
            "apply_error": str(exc)[:500],
            "pass": False,
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

    print("GB10_MINIMAX_M3_PARITY SELFTEST " + ("PASS" if ok else "FAIL")
          + " (no hardware parity claim)")
    return 0 if ok else 1


# ── main ───────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="receipt path")
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
              f"cosine_silu={ent['cosine_silu_reference']}")
        verdict = verdict and ent["pass"]

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
    print(f"case D: {receipt['case_d']}")
    verdict = verdict and receipt["case_d"]["pass"]

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
