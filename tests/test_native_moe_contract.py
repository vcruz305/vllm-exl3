"""CPU metadata checks and CUDA numerical checks for the extended MoE ABI."""

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import vllm_exl3.exl3 as exl3


@pytest.mark.parametrize("intermediate", [1024, 2048])
@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("limit", [None, 0.0, 10.0])
def test_native_geometry_accepts_tp1_tp2_and_clipping(intermediate, bits, limit):
    # This predicate inspects metadata only; do not pretend to run CUDA on CPU.
    x = SimpleNamespace(shape=(8, 4096), is_cuda=True, dim=lambda: 2)
    layer = SimpleNamespace(_exl3_intermediate_local=intermediate, _exl3_k=bits)
    assert exl3._native_moe_dimensions_supported(x, layer, [{}], limit)


@pytest.mark.parametrize(
    "shape,cuda,intermediate,bits,limit",
    [((1, 4096), False, 2048, 4, None),
     ((4096,), True, 2048, 4, None),
     ((0, 4096), True, 2048, 4, None),
     ((9, 4096), True, 2048, 4, None),
     ((1, 2048), True, 2048, 4, None),
     ((1, 4096), True, 1536, 4, None),
     ((1, 4096), True, 2048, 5, None),
     ((1, 4096), True, 2048, 4, -1.0),
     ((1, 4096), True, 2048, 4, math.nan),
     ((1, 4096), True, 2048, 4, math.inf)],
)
def test_native_geometry_retains_unsupported_guards(shape, cuda, intermediate, bits, limit):
    x = SimpleNamespace(shape=shape, is_cuda=cuda, dim=lambda: len(shape))
    layer = SimpleNamespace(_exl3_intermediate_local=intermediate, _exl3_k=bits)
    assert not exl3._native_moe_dimensions_supported(x, layer, [{}], limit)


@pytest.fixture
def native_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for native MoE numerical checks")
    extension = pytest.importorskip("vllm_exl3_c")
    assert getattr(extension, "P2B_MOE_ABI_VERSION", 1) >= 2, "rebuild vllm_exl3_c for MoE ABI 2"
    return extension


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("intermediate", [1024, 2048])
def test_native_moe_clipping_width_and_graph_parity(native_cuda, bits, intermediate, monkeypatch):
    """Independent CPU reconstruction + dense matmul, then eager and graph replay.

    The CPU decoder uses a separate implementation from the CUDA GEMV tile.
    This checks the full result, including scale, rather than cosine alone.
    It is a small synthetic fixture, not a full-model TP2/DeepSeek qualification.
    """
    device = torch.device("cuda")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(17)
    hidden = 4096
    num_experts = 2
    top_k = 6 if bits in (2, 3) else 8

    def make_projection(in_features, out_features, output_scale):
        packed, suh, svh, dense = [], [], [], []
        for _ in range(num_experts):
            t = torch.randint(-32768, 32767, (in_features // 16, out_features // 16, 16 * bits), dtype=torch.int16)
            u = torch.full((in_features,), 1.0 / 64, dtype=torch.float16)
            v = torch.full((out_features,), output_scale, dtype=torch.float16)
            # Deliberately reconstruct on CPU, outside the CUDA fast path.
            dense.append(native_cuda.dequant_trellis(t, u, v, bits, True).to(device).float())
            packed.append(t.to(device))
            suh.append(u.to(device))
            svh.append(v.to(device))
        tables = [torch.tensor([t.data_ptr() for t in values], device=device, dtype=torch.int64)
                  for values in (packed, suh, svh)]
        return (packed, suh, svh), tables, dense

    gate_storage, gate_ptrs, gate_dense = make_projection(hidden, intermediate, 8.0)
    up_storage, up_ptrs, up_dense = make_projection(hidden, intermediate, 8.0)
    down_storage, down_ptrs, down_dense = make_projection(intermediate, hidden, 1.0)
    # Retain pointees through the last graph replay.
    storage = (gate_storage, up_storage, down_storage)
    ptrs = gate_ptrs + up_ptrs + down_ptrs
    # Repeated routes exercise reduction over all six/eight entries with only
    # two allocated experts; the last route has zero weight.
    ids = torch.arange(top_k, device=device, dtype=torch.int32).remainder(num_experts)
    weights = torch.linspace(0.1, 0.9, top_k, device=device, dtype=torch.float16)
    weights[-1] = 0
    weights /= weights.sum()
    x = torch.randn(1, hidden, device=device, dtype=torch.float16)
    out = torch.empty_like(x)

    def reference(limit):
        result = torch.zeros_like(x, dtype=torch.float32)
        saturated = False
        for e in range(num_experts):
            g = (x.float() @ gate_dense[e]).half().float()
            u = (x.float() @ up_dense[e]).half().float()
            if limit:
                saturated |= bool(((g > limit) | (u.abs() > limit)).any())
                g = g.clamp(max=limit)
                u = u.clamp(min=-limit, max=limit)
            h = (torch.nn.functional.silu(g) * u).half()
            d = (h.float() @ down_dense[e]).half().float()
            result += d * weights[ids == e].float().sum()
        if limit:
            assert saturated, "fixture must exercise clipping, not only pass a limit"
        return result.half().float()

    def check(expected):
        actual = out.float()
        assert torch.isfinite(actual).all()
        relative_error = (actual - expected).norm() / expected.norm().clamp_min(1e-8)
        assert relative_error < 0.01, f"relative error: {relative_error.item()}"
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01 * expected.abs().max().item())

    # Limit 1 makes pre-/post-SiLU clipping substantially different; limit 10
    # covers the GLM/DeepSeek setting. Zero protects the original plain path.
    for limit in (0.0, 1.0, 10.0):
        def run():
            native_cuda.p2b_fused_moe(x, out, *ptrs, ids, weights, bits, bits, bits, True, intermediate, limit)

        expected = reference(limit)
        run()
        torch.cuda.synchronize()
        check(expected)
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        # Replayed graphs must consume changed inputs, not capture-time values.
        for _ in range(2):
            x.copy_(torch.randn_like(x))
            expected = reference(limit)
            graph.replay()
            torch.cuda.synchronize()
            check(expected)
        del graph
    assert storage


@pytest.mark.parametrize("bad", ["rows", "width", "negative_limit", "nan_limit", "strided_pointers"])
def test_native_moe_rejects_unsafe_calls_before_launch(native_cuda, bad):
    x = torch.zeros(1, 4096, dtype=torch.float16, device="cuda")
    out = torch.empty_like(x)
    # Null pointees are intentional: argument validation must fail before launch.
    ptrs = [torch.zeros(2, dtype=torch.int64, device="cuda") for _ in range(9)]
    ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    weights = torch.ones(1, dtype=torch.float16, device="cuda")
    width, limit = 2048, 0.0
    if bad == "rows":
        x = x.repeat(2, 1)
    elif bad == "width":
        width = 1536
    elif bad == "negative_limit":
        limit = -1.0
    elif bad == "nan_limit":
        limit = math.nan
    else:
        ptrs[0] = torch.zeros(4, dtype=torch.int64, device="cuda")[::2]
    with pytest.raises(RuntimeError, match="fused MoE"):
        native_cuda.p2b_fused_moe(x, out, *ptrs, ids, weights, 4, 4, 4, True, width, limit)
