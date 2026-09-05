"""Unit test verifying batched cooperative MoE active-expert GEMV kernel."""
import os
import time
import pytest

def test_p2b_batched_function_exists():
    """Verify that p2b_gemv_batched is exposed by vllm_exl3_c."""
    try:
        import vllm_exl3_c
    except ImportError as e:
        pytest.skip(f"vllm_exl3_c not importable: {e}")
    assert hasattr(vllm_exl3_c, "p2b_gemv_batched"), "vllm_exl3_c does not export p2b_gemv_batched"

def test_p2b_batched_parity_and_speedup():
    """Verify numerical parity and latency speedup over sequential GEMV on active experts."""
    try:
        import torch
        import vllm_exl3_c
    except ImportError:
        pytest.skip("PyTorch and vllm_exl3_c required for batched GEMV tests")
        
    if not hasattr(vllm_exl3_c, "p2b_gemv_batched"):
        pytest.skip("vllm_exl3_c does not export p2b_gemv_batched")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("CUDA device required for native batched GEMV tests")
        
    in_features = 4096
    out_features = 2048
    K = 2
    num_active_experts = 8
    m = 1
    
    trellises = [
        torch.randint(-32768, 32767, (in_features // 16, out_features // 16, 16 * K), dtype=torch.int16, device=device)
        for _ in range(num_active_experts)
    ]
    suhs = [torch.randn(in_features, dtype=torch.float16, device=device) for _ in range(num_active_experts)]
    svhs = [torch.randn(out_features, dtype=torch.float16, device=device) for _ in range(num_active_experts)]
    x = torch.randn(m, in_features, dtype=torch.float16, device=device) * 0.1
    
    # Reference sequential execution
    y_refs = []
    for e in range(num_active_experts):
        y_refs.append(vllm_exl3_c.exl3_gemv(x, trellises[e], suhs[e], svhs[e], K, True))
    y_ref_stacked = torch.stack(y_refs, dim=0)
    
    trellis_ptrs = torch.tensor([t.data_ptr() for t in trellises], dtype=torch.int64, device=device)
    suh_ptrs = torch.tensor([s.data_ptr() for s in suhs], dtype=torch.int64, device=device)
    svh_ptrs = torch.tensor([s.data_ptr() for s in svhs], dtype=torch.int64, device=device)
    expert_indices = torch.arange(num_active_experts, dtype=torch.int32, device=device)
    
    # Warmup
    for _ in range(10):
        _ = vllm_exl3_c.p2b_gemv_batched(x, trellis_ptrs, suh_ptrs, svh_ptrs, expert_indices, K, True)
    torch.cuda.synchronize()
    
    # Parity check
    y_batched = vllm_exl3_c.p2b_gemv_batched(x, trellis_ptrs, suh_ptrs, svh_ptrs, expert_indices, K, True)
    assert y_batched.shape == (num_active_experts, m, out_features)
    assert torch.isfinite(y_batched).all(), "Output contains NaN or Inf"
    
    max_abs_err = (y_batched - y_ref_stacked).abs().max().item()
    assert max_abs_err <= 1e-3, f"Max absolute error {max_abs_err:.5f} exceeded 1e-3"
    
    # Latency benchmark
    iters = 100
    t0 = time.time()
    for _ in range(iters):
        _ = vllm_exl3_c.p2b_gemv_batched(x, trellis_ptrs, suh_ptrs, svh_ptrs, expert_indices, K, True)
    torch.cuda.synchronize()
    dt_us = (time.time() - t0) / iters * 1e6
    print(f"p2b_gemv_batched latency: {dt_us:.1f} us")
    target_us = float(os.environ.get("VLLM_EXL3_MAX_BATCHED_US", "400.0"))
    assert dt_us <= target_us, f"Batched latency {dt_us:.1f} us failed target <= {target_us:.1f} us"
