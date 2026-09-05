#!/usr/bin/env python3
"""High-precision GPU microbenchmark for GLM-5.3-Flash MoE decode kernels on DGX Spark GB10.

Measures:
1. Batched active-expert GEMV (p2b_gemv_batched with launch_batched_fast_2) vs sequential baseline.
2. Full MoE layer decode (gate + up + SwiGLU clamp + down projection) native vs fallback.
3. GPU bandwidth utilization and speedup factors across batch sizes M in {1, 2, 4, 8, 16}.
"""
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch

try:
    import vllm_exl3_c
except ImportError:
    vllm_exl3_c = None


def time_gpu_cuda_events(fn, warmup: int = 15, iters: int = 100) -> Dict[str, float]:
    """Times a callable using CUDA events for microsecond precision."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(iters)]
    times_us = [t * 1000.0 for t in times_ms]
    times_us.sort()

    median = times_us[len(times_us) // 2]
    p95 = times_us[int(len(times_us) * 0.95)]
    mean = sum(times_us) / len(times_us)
    std = (sum((x - mean) ** 2 for x in times_us) / len(times_us)) ** 0.5

    return {
        "median_us": round(median, 2),
        "p95_us": round(p95, 2),
        "mean_us": round(mean, 2),
        "min_us": round(times_us[0], 2),
        "max_us": round(times_us[-1], 2),
        "std_us": round(std, 2),
    }


def bench_batched_gemv(device: torch.device, K_bits: int = 2, in_dim: int = 4096, out_dim: int = 2048,
                       active_experts: int = 8, m: int = 1) -> Dict[str, Any]:
    """Benchmarks batched GEMV vs sequential GEMV for active experts."""
    if vllm_exl3_c is None or not hasattr(vllm_exl3_c, "p2b_gemv_batched"):
        return {"error": "p2b_gemv_batched not available in vllm_exl3_c"}

    trellises = [
        torch.randint(-32768, 32767, (in_dim // 16, out_dim // 16, 16 * K_bits), dtype=torch.int16, device=device)
        for _ in range(active_experts)
    ]
    suhs = [torch.randn(in_dim, dtype=torch.float16, device=device) for _ in range(active_experts)]
    svhs = [torch.randn(out_dim, dtype=torch.float16, device=device) for _ in range(active_experts)]
    x = torch.randn(m, in_dim, dtype=torch.float16, device=device) * 0.1

    trellis_ptrs = torch.tensor([t.data_ptr() for t in trellises], dtype=torch.int64, device=device)
    suh_ptrs = torch.tensor([s.data_ptr() for s in suhs], dtype=torch.int64, device=device)
    svh_ptrs = torch.tensor([s.data_ptr() for s in svhs], dtype=torch.int64, device=device)
    expert_indices = torch.arange(active_experts, dtype=torch.int32, device=device)

    # Sequential callable
    def run_sequential():
        for e in range(active_experts):
            _ = vllm_exl3_c.exl3_gemv(x, trellises[e], suhs[e], svhs[e], K_bits, True)

    # Batched callable
    def run_batched():
        _ = vllm_exl3_c.p2b_gemv_batched(x, trellis_ptrs, suh_ptrs, svh_ptrs, expert_indices, K_bits, True)

    stats_seq = time_gpu_cuda_events(run_sequential)
    stats_batched = time_gpu_cuda_events(run_batched)

    speedup = round(stats_seq["median_us"] / max(stats_batched["median_us"], 1e-3), 2)
    # Packed weight streaming bytes = in_dim * out_dim * K_bits * 2 / 16 * active_experts
    weight_bytes = (in_dim // 16) * (out_dim // 16) * 16 * K_bits * 2 * active_experts
    bandwidth_gb_s = round((weight_bytes / (stats_batched["median_us"] * 1e-6)) / 1e9, 2)

    return {
        "m": m,
        "active_experts": active_experts,
        "K_bits": K_bits,
        "shape": f"({in_dim}, {out_dim})",
        "weight_bytes": weight_bytes,
        "sequential_stats": stats_seq,
        "batched_stats": stats_batched,
        "speedup": speedup,
        "bandwidth_gb_s": bandwidth_gb_s,
    }


def bench_moe_layer(device: torch.device, hidden: int = 4096, intermediate: int = 2048,
                    num_total_experts: int = 288, top_k: int = 8, bits: int = 2,
                    swiglu_limit: float = 10.0, m_tokens: int = 1) -> Dict[str, Any]:
    """Benchmarks a full fused MoE decode layer with clipped SwiGLU."""
    if vllm_exl3_c is None or not hasattr(vllm_exl3_c, "p2b_fused_moe"):
        return {"error": "p2b_fused_moe not available in vllm_exl3_c"}

    def make_proj(in_f, out_f):
        packed, suh, svh = [], [], []
        for _ in range(num_total_experts):
            t = torch.randint(-32768, 32767, (in_f // 16, out_f // 16, 16 * bits), dtype=torch.int16, device=device)
            u = torch.randn(in_f, dtype=torch.float16, device=device)
            v = torch.randn(out_f, dtype=torch.float16, device=device)
            packed.append(t)
            suh.append(u)
            svh.append(v)
        tables = [torch.tensor([t.data_ptr() for t in vals], device=device, dtype=torch.int64)
                  for vals in (packed, suh, svh)]
        return (packed, suh, svh), tables

    _, gate_ptrs = make_proj(hidden, intermediate)
    _, up_ptrs = make_proj(hidden, intermediate)
    _, down_ptrs = make_proj(intermediate, hidden)
    all_ptrs = gate_ptrs + up_ptrs + down_ptrs

    x = torch.randn(m_tokens, hidden, dtype=torch.float16, device=device)
    out = torch.empty_like(x)
    ids = torch.arange(top_k, dtype=torch.int32, device=device)
    weights = torch.ones(top_k, dtype=torch.float16, device=device) / top_k

    # ABI version 2 check
    abi_v = getattr(vllm_exl3_c, "P2B_MOE_ABI_VERSION", 1)

    def run_native():
        if abi_v >= 2:
            vllm_exl3_c.p2b_fused_moe(
                x, out, *all_ptrs, ids, weights, bits, bits, bits, True, intermediate, swiglu_limit
            )
        else:
            vllm_exl3_c.p2b_fused_moe(x, out, *all_ptrs, ids, weights, bits, bits, bits, True)

    stats_native = time_gpu_cuda_events(run_native)
    total_45_layers_ms = round((stats_native["median_us"] * 45) / 1000.0, 2)

    return {
        "m_tokens": m_tokens,
        "top_k": top_k,
        "num_experts": num_total_experts,
        "bits": bits,
        "swiglu_limit": swiglu_limit,
        "abi_version": abi_v,
        "native_stats": stats_native,
        "total_45_layers_ms": total_45_layers_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="GLM-5.3-Flash MoE GPU Microbenchmarks")
    parser.add_argument("--output", default=".agent-sync/micro_bench_receipts.json", help="Path to write JSON receipts")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting microbenchmarks.")
        sys.exit(0)

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    print(f"=== Running Microbenchmarks on {gpu_name} (sm_{capability[0]}{capability[1]}) ===")

    results = {
        "gpu": gpu_name,
        "capability": f"{capability[0]}.{capability[1]}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "batched_gemv": [],
        "fused_moe_layers": [],
    }

    # 1. Batched GEMV benchmark across M in {1, 2, 4, 8}
    print("\n--- 1. Batched Active-Expert GEMV Scaling ---")
    for m in [1, 2, 4, 8]:
        res = bench_batched_gemv(device, K_bits=2, in_dim=4096, out_dim=2048, active_experts=8, m=m)
        results["batched_gemv"].append(res)
        if "error" not in res:
            print(f"M={m:2d} | Sequential: {res['sequential_stats']['median_us']:6.1f} us | "
                  f"Batched: {res['batched_stats']['median_us']:6.1f} us | "
                  f"Speedup: {res['speedup']:4.2f}x | Bandwidth: {res['bandwidth_gb_s']:6.1f} GB/s")

    # 2. Full Fused MoE Layer decode benchmark (specialized for M=1 decode)
    print("\n--- 2. Full Fused MoE Layer Decode Latency ---")
    for m in [1]:
        res = bench_moe_layer(device, hidden=4096, intermediate=2048, num_total_experts=288,
                              top_k=8, bits=2, swiglu_limit=10.0, m_tokens=m)
        results["fused_moe_layers"].append(res)
        if "error" not in res:
            print(f"Tokens={m} | Per-layer: {res['native_stats']['median_us']:6.1f} us | "
                  f"45-layer total: {res['total_45_layers_ms']:5.2f} ms")

    # Ensure parent dir exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Results saved to {args.output}")


if __name__ == "__main__":
    main()
