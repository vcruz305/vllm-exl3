"""Unit tests for Ornith-1.5 / Qwen3.5-MoE Iteration 2 optimizations."""

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys

import pytest
import torch

import vllm_exl3
import vllm_exl3.exl3 as exl3

vllm_exl3.register()

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_SCAN = os.path.join(_ROOT, "tools", "exl3_pack_tools", "qwen_pack_scan.py")
_CFG = os.path.join(_ROOT, "tools", "exl3_pack_tools", "qwen_pack_config.py")


def _write_safetensors_hdr_only(path: str, meta: dict[str, tuple[str, list[int]]]) -> None:
    hdr = {}
    off = 0
    dtype_bytes = {"I16": 2, "F16": 2, "BF16": 2, "I32": 4}
    for name, (dt, shape) in meta.items():
        n = 1
        for s in shape:
            n *= s
        b = n * dtype_bytes[dt]
        hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + b]}
        off += b
    raw = json.dumps(hdr).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)


def test_pack_scan_and_config_mtp_mul1_head_vision(tmp_path):
    meta = {
        "lm_head.trellis": ("I16", [192, 15520, 128]),
        "lm_head.suh": ("F16", [3072]),
        "lm_head.svh": ("F16", [248320]),
        "lm_head.mul1": ("I32", []),
        "model.language_model.layers.0.mlp.experts.0.gate_proj.trellis": ("I16", [192, 64, 48]),
        "model.language_model.layers.0.mlp.experts.0.gate_proj.mul1": ("I32", []),
        "model.language_model.layers.0.mlp.experts.0.up_proj.trellis": ("I16", [192, 64, 48]),
        "model.language_model.layers.0.mlp.experts.0.down_proj.trellis": ("I16", [64, 192, 48]),
        "mtp.fc.trellis": ("I16", [384, 192, 32]),
        "mtp.fc.mul1": ("I32", []),
        "mtp.layers.0.self_attn.q_proj.trellis": ("I16", [192, 384, 32]),
        "mtp.layers.0.mlp.experts.0.gate_proj.trellis": ("I16", [192, 64, 32]),
        "mtp.layers.0.mlp.experts.0.up_proj.trellis": ("I16", [192, 64, 32]),
        "mtp.layers.0.mlp.experts.0.down_proj.trellis": ("I16", [64, 192, 32]),
        "model.visual.blocks.0.attn.qkv.weight": ("BF16", [3456, 1152]),
        "model.visual.blocks.0.attn.q_proj.trellis": ("I16", [72, 72, 64]),
        "model.visual.blocks.0.attn.proj.trellis": ("I16", [72, 72, 64]),
        "model.visual.blocks.0.mlp.linear_fc1.trellis": ("I16", [72, 272, 64]),
        "model.visual.merger.linear_fc1.trellis": ("I16", [288, 288, 64]),
    }
    _write_safetensors_hdr_only(str(tmp_path / "model-00001-of-00001.safetensors"), meta)
    cfg = {
        "text_config": {"num_hidden_layers": 48},
        "quantization_config": {"quant_method": "exl3", "bits": 2.58},
    }
    json.dump(cfg, open(tmp_path / "config.json", "w"))

    subprocess.run([sys.executable, _SCAN, str(tmp_path)], check=True)
    scan = json.load(open(tmp_path / "pack_scan.json"))
    assert scan["expert_k_nonuniform"] == []
    assert scan["expert_k_per_layer"] == {"0": {"down_proj": [3], "gate_proj": [3], "up_proj": [3]}}
    assert scan["mtp_expert_k_per_layer"] == {"0": {"down_proj": [2], "gate_proj": [2], "up_proj": [2]}}

    subprocess.run([sys.executable, _CFG, str(tmp_path)], check=True)
    out_cfg = json.load(open(tmp_path / "config.json"))["quantization_config"]
    assert out_cfg["codebook"] == "mul1"
    assert out_cfg["head_bits"] == 8
    assert out_cfg["layer_bits"]["mtp.layers.0"] == 2
    assert out_cfg["layer_bits"]["48"] == 2
    layers = out_cfg["non_routed_exl3"]["layers"]
    assert layers["mtp.fc"]["bits"] == 2
    assert layers["model.fc"]["bits"] == 2
    assert layers["mtp.layers.0.self_attn.qkv_proj"]["bits"] == 2
    assert layers["mtp.layers.48.self_attn.qkv_proj"]["bits"] == 2
    assert "visual.blocks.0.attn.qkv" not in layers
    assert "visual.blocks.0.attn.qkv_proj" not in layers
    assert "visual.blocks.0.attn.q_proj" not in layers
    assert layers["visual.blocks.0.attn.proj"]["bits"] == 4
    assert layers["visual.blocks.0.mlp.linear_fc1"]["bits"] == 4
    assert layers["visual.merger.linear_fc1"]["bits"] == 4

    exl3_cfg = exl3.Exl3Config.from_config(out_cfg)
    assert exl3_cfg.bits_for_prefix("mtp.layers.0.mlp.experts") == 2
    assert exl3_cfg.bits_for_prefix("mtp.layers.48.mlp.experts") == 2
    assert exl3_cfg.bits_for_prefix("model.layers.0.mlp.experts") == 3
    assert not exl3_cfg._matches_non_routed_exl3("visual.blocks.0.attn.qkv")
    assert exl3_cfg._matches_non_routed_exl3("visual.blocks.0.attn.proj")


def test_asymmetric_uniform_moe_coop_and_shard_coalescing_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    try:
        import exllamav3_ext  # noqa: F401
    except ImportError:
        pytest.skip("exllamav3_ext required")
    import vllm.distributed as dist

    dist.get_tensor_model_parallel_rank = lambda: 0
    dist.get_tensor_model_parallel_world_size = lambda: 1

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(2026)

    # 1. Test asymmetric (kg=2, ku=2, kd=3) MoE layer with _COOP=1 vs _COOP=0 and tokens*topk > 256
    hidden, inter, n_exp, topk = 256, 256, 8, 8
    cfg = exl3.Exl3Config(bits=2, codebook="mul1")
    moe_cfg = type(
        "_MC",
        (),
        {
            "hidden_dim": hidden,
            "num_experts": n_exp,
            "num_local_experts": n_exp,
            "experts_per_token": topk,
            "activation": "silu",
            "rocm_aiter_fmoe_enabled": False,
            "swiglu_limit": None,
        },
    )()
    method = object.__new__(exl3.Exl3MoEMethod)
    method.moe = moe_cfg
    method.quant_config = cfg
    method.bits = 2
    method._logged = False

    layer = torch.nn.Module()
    layer.moe_config = moe_cfg
    layer.layer_name = "model.layers.30.mlp.experts"
    layer.global_num_experts = n_exp
    layer.local_num_experts = n_exp
    layer.starting_expert_offset = 0
    layer._map_global_expert_id_to_local_expert_id = lambda eid: int(eid)
    method.create_weights(layer, n_exp, hidden, inter, torch.bfloat16)
    for name in ("w13_suh", "w13_svh", "w13_mcg", "w13_mul1", "w2_suh", "w2_svh", "w2_mcg", "w2_mul1"):
        getattr(layer, name).data = getattr(layer, name).data.to(device)

    for e in range(n_exp):
        gt = torch.randint(-32768, 32767, (hidden // 16, inter // 16, 32), dtype=torch.int16, device=device, generator=gen)
        ut = torch.randint(-32768, 32767, (hidden // 16, inter // 16, 32), dtype=torch.int16, device=device, generator=gen)
        dt = torch.randint(-32768, 32767, (inter // 16, hidden // 16, 48), dtype=torch.int16, device=device, generator=gen)
        method._load_exl3(layer.w13_trellis, gt, "w1.trellis", "w1", e, return_success=True)
        method._load_exl3(layer.w13_trellis, ut, "w3.trellis", "w3", e, return_success=True)
        method._load_exl3(layer.w2_trellis, dt, "w2.trellis", "w2", e, return_success=True)
        for p, sh, dim in (("w13", "w1", hidden), ("w13", "w3", hidden), ("w2", "w2", inter)):
            s = (torch.randn(dim, generator=gen, device=device).sign() / math.sqrt(dim)).half()
            method._load_exl3(getattr(layer, f"{p}_suh"), s, f"{sh}.suh", sh, e, return_success=True)
        for p, sh, dim in (("w13", "w1", inter), ("w13", "w3", inter), ("w2", "w2", hidden)):
            s = (torch.randn(dim, generator=gen, device=device).sign() / math.sqrt(dim)).half()
            method._load_exl3(getattr(layer, f"{p}_svh"), s, f"{sh}.svh", sh, e, return_success=True)
        m_t = torch.tensor(exl3.MUL1_MARKER_SIGNED_INT32, dtype=torch.int32, device=device)
        method._load_exl3(layer.w13_mul1, m_t, "w1.mul1", "w1", e, return_success=True)
        method._load_exl3(layer.w13_mul1, m_t, "w3.mul1", "w3", e, return_success=True)
        method._load_exl3(layer.w2_mul1, m_t, "w2.mul1", "w2", e, return_success=True)

    method.process_weights_after_loading(layer)
    assert layer._exl3_k_tuple == (2, 2, 3)
    assert layer._exl3_coop_compat is True
    assert torch.all(layer._exl3_K_gate_arr == 4)
    assert torch.all(layer._exl3_K_up_arr == 4)
    assert torch.all(layer._exl3_K_down_arr == 6)

    for bsz in (1, 2, 4, 8, 1, 3, 1, 40):
        x = torch.randn(bsz, hidden, generator=gen, device=device, dtype=torch.bfloat16)
        ids = torch.arange(topk, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, topk).contiguous()
        rw = torch.softmax(torch.randn(bsz, topk, generator=gen, device=device, dtype=torch.float32), dim=-1)
        ref = exl3.apply_exl3_python_loop(x, ids, rw, layer._exl3_inners, None, None)
        exl3._COOP = True
        y_coop = exl3.apply_exl3_fused_moe(x, ids, rw, layer, layer._exl3_inners, None, None)
        exl3._COOP = False
        y_nocoop = exl3.apply_exl3_fused_moe(x, ids, rw, layer, layer._exl3_inners, None, None)
        exl3._COOP = True
        err_coop_vs_moe = (y_coop.float() - y_nocoop.float()).norm() / y_nocoop.float().norm()
        err_coop_vs_ref = (y_coop.float() - ref.float()).norm() / ref.float().norm()
        assert err_coop_vs_moe < 2e-3, f"bsz={bsz} coop_vs_moe err={err_coop_vs_moe.item()}"
        assert err_coop_vs_ref < 2e-2, f"bsz={bsz} coop_vs_ref err={err_coop_vs_ref.item()}"

    # 2. Test dense shard coalescing on a 4-shard in_proj_qkvz where shards (0, 1, 2) come from in_proj_qkv
    lin_cfg = exl3.Exl3Config(bits=6, codebook="mul1")
    lin_method = exl3.Exl3LinearMethod(lin_cfg, bits=6)
    shards = [256, 256, 512, 512]
    lin_layer = torch.nn.Module()
    lin_method.create_weights(lin_layer, hidden, shards, hidden, sum(shards), torch.bfloat16)
    for name in ("trellis", "suh", "svh", "mcg", "mul1"):
        getattr(lin_layer, name).data = getattr(lin_layer, name).data.to(device)

    t_qkv = torch.randint(-32768, 32767, (hidden // 16, 1024 // 16, 96), dtype=torch.int16, device=device, generator=gen)
    suh_qkv = (torch.randn(hidden, generator=gen, device=device).sign() / math.sqrt(hidden)).half()
    svh_qkv = (torch.randn(1024, generator=gen, device=device).sign() / math.sqrt(1024)).half()
    t_z = torch.randint(-32768, 32767, (hidden // 16, 512 // 16, 96), dtype=torch.int16, device=device, generator=gen)
    suh_z = (torch.randn(hidden, generator=gen, device=device).sign() / math.sqrt(hidden)).half()
    svh_z = (torch.randn(512, generator=gen, device=device).sign() / math.sqrt(512)).half()
    mul1_m = torch.tensor([exl3.MUL1_MARKER_SIGNED_INT32], dtype=torch.int32, device=device)

    for name, t in (("trellis", t_qkv), ("suh", suh_qkv), ("svh", svh_qkv), ("mul1", mul1_m)):
        p = getattr(lin_layer, name)
        p.weight_loader(p, t, (0, 1, 2))
    for name, t in (("trellis", t_z), ("suh", suh_z), ("svh", svh_z), ("mul1", mul1_m)):
        p = getattr(lin_layer, name)
        p.weight_loader(p, t, 3)

    lin_method.process_weights_after_loading(lin_layer)
    assert len(lin_layer._exl3_linears) == 2
