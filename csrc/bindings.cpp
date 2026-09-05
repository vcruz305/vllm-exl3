#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <vector>

#include "exl3_dequant.cuh"
#include "p2b_batched.cuh"
#include "p2b_moe.cuh"
#include "exl3_gemm.cuh"
#include "exl3_fat_gemm.cuh"

namespace {

at::Tensor dequant_cpu(const at::Tensor& trellis, const at::Tensor& suh,
                       const at::Tensor& svh, int64_t bits, bool mcg) {
    const auto packed = trellis.contiguous();
    const auto su = suh.to(at::kCPU).contiguous();
    const auto sv = svh.to(at::kCPU).contiguous();
    const int64_t rows = packed.size(0) * 16;
    const int64_t cols = packed.size(1) * 16;
    TORCH_CHECK(rows % 128 == 0 && cols % 128 == 0,
                "trellis dimensions must reconstruct whole 128-element Hadamard blocks");
    auto out = at::empty({rows, cols}, packed.options().dtype(at::kHalf));
    auto* dst = out.data_ptr<c10::Half>();
    const auto* p = packed.data_ptr<int16_t>();
    const auto* sup = su.data_ptr<c10::Half>();
    const auto* svp = sv.data_ptr<c10::Half>();
    const std::size_t words = static_cast<std::size_t>(16 * bits);
    std::vector<float> matrix(static_cast<std::size_t>(rows * cols), 0.0f);

    // Each packed 16*K tile is decoded by eight logical warps in the native
    // kernel.  A warp's 32 lanes produce eight values each; the lane+4
    // shuffle then interleaves those values into the 16x16 row-major tile.
    // Reproduce that tensor-core fragment swizzle on the host so the extension
    // has the same bit layout as reconstruct_had_slice.
    for (int64_t kt = 0; kt < packed.size(0); ++kt) {
        for (int64_t nt = 0; nt < packed.size(1); ++nt) {
            const auto* tile = reinterpret_cast<const std::uint16_t*>(
                p + (kt * packed.size(1) + nt) * words);
            std::array<std::array<std::uint16_t, 8>, 32> lane{};
            for (int l = 0; l < 32; ++l) {
                for (int j = 0; j < 8; ++j) {
                    const std::size_t start = (static_cast<std::size_t>(l * 8 + j) + 257u) *
                                              static_cast<int>(bits) - 16u;
                    const auto window = vllm_exl3::read_window(tile, words, start);
                    lane[l][j] = vllm_exl3::decode_codebook_bits(
                        window, mcg ? 1 : 2);
                }
            }
            for (int l = 0; l < 32; ++l) {
                if (l & 4) continue;
                const int row0 = (l % 4) * 2;
                const int row2 = row0 + 8;
                const int col0 = (l / 8) * 2;
                const int col4 = col0 + 8;
                const auto emit = [&](int row, int col, int a, int b) {
                    matrix[static_cast<std::size_t>((kt * 16 + row) * cols +
                                                     nt * 16 + col)] =
                        vllm_exl3::half_bits_to_float(a);
                    matrix[static_cast<std::size_t>((kt * 16 + row) * cols +
                                                     nt * 16 + col + 1)] =
                        vllm_exl3::half_bits_to_float(b);
                };
                const auto& own = lane[l];
                const auto& peer = lane[l + 4];
                emit(row0, col0, own[0], peer[0]);
                emit(row0 + 1, col0, own[1], peer[1]);
                emit(row2, col0, own[2], peer[2]);
                emit(row2 + 1, col0, own[3], peer[3]);
                emit(row0, col4, own[4], peer[4]);
                emit(row0 + 1, col4, own[5], peer[5]);
                emit(row2, col4, own[6], peer[6]);
                emit(row2 + 1, col4, own[7], peer[7]);
            }
        }
    }

    constexpr float norm = 0.08838834764831845f;
    std::array<float, 128> line{};
    for (int64_t r0 = 0; r0 < rows; r0 += 128) {
        for (int64_t c0 = 0; c0 < cols; c0 += 128) {
            for (int r = 0; r < 128; ++r) {
                for (int c = 0; c < 128; ++c) line[c] = matrix[(r0 + r) * cols + c0 + c];
                vllm_exl3::hadamard(line.data(), 128);
                for (int c = 0; c < 128; ++c) matrix[(r0 + r) * cols + c0 + c] = line[c] * norm;
            }
            for (int c = 0; c < 128; ++c) {
                for (int r = 0; r < 128; ++r) line[r] = matrix[(r0 + r) * cols + c0 + c];
                vllm_exl3::hadamard(line.data(), 128);
                for (int r = 0; r < 128; ++r) {
                    const float value = line[r] * norm * static_cast<float>(sup[r0 + r]) *
                                         static_cast<float>(svp[c0 + c]);
                    dst[(r0 + r) * cols + c0 + c] = c10::Half(value);
                }
            }
        }
    }
    return out;
}

}  // namespace

// CUDA implementation is compiled from exl3_gemv.cu.  Keeping the declaration
// here also lets CPU-only builds retain a functional reference fallback.
at::Tensor exl3_gemv_cuda(const at::Tensor& x, const at::Tensor& trellis,
                          const at::Tensor& suh, const at::Tensor& svh,
                          int64_t bits, bool mcg, int64_t mmode);

at::Tensor exl3_gemm(const at::Tensor& x, const at::Tensor& trellis,
                     const at::Tensor& suh, const at::Tensor& svh,
                     int64_t bits, bool mcg) {
    return exl3_gemm_cuda(x.contiguous().view({x.numel() / x.size(-1), x.size(-1)}),
                          trellis, suh, svh, bits, mcg);
}

at::Tensor dequant_trellis(const at::Tensor& trellis, const at::Tensor& suh,
                           const at::Tensor& svh, int64_t bits, bool mcg) {
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4 || bits == 8,
                "K must be one of 2, 3, 4, or 8");
    TORCH_CHECK(trellis.dim() == 3 && trellis.scalar_type() == at::kShort,
                "trellis must be int16 [rows/16, cols/16, 16*K]");
    TORCH_CHECK(trellis.size(2) == 16 * bits, "trellis last dimension must be 16*K");
    TORCH_CHECK(trellis.size(0) > 0 && trellis.size(1) > 0, "trellis dimensions must be positive");
    TORCH_CHECK(suh.scalar_type() == at::kHalf && svh.scalar_type() == at::kHalf,
                "suh and svh must be float16");
    TORCH_CHECK(suh.numel() >= trellis.size(0) * 16 && svh.numel() >= trellis.size(1) * 16,
                "suh/svh are too short");
    const auto device = trellis.device();
    auto result = dequant_cpu(trellis.to(at::kCPU), suh, svh, bits, mcg);
    return device.is_cpu() ? result : result.to(device);
}

at::Tensor exl3_gemv(const at::Tensor& x, const at::Tensor& trellis,
                     const at::Tensor& suh, const at::Tensor& svh,
                     int64_t bits, bool mcg, int64_t mmode) {
    TORCH_CHECK(x.dim() >= 2, "x must have shape [..., in_features]");
    const int64_t m = x.numel() / x.size(-1);
    TORCH_CHECK(m >= 1 && m <= 8, "native GEMV supports 1 <= m <= 8");
    TORCH_CHECK(x.size(-1) == trellis.size(0) * 16,
                "x feature dimension does not match trellis");
    TORCH_CHECK(mmode == 0 || mmode == 1, "mmode must be 0 or 1");
    auto x2 = x.contiguous().view({m, x.size(-1)});
    if (x.is_cuda()) return exl3_gemv_cuda(x2, trellis, suh, svh, bits, mcg, mmode);
    auto w = dequant_trellis(trellis, suh, svh, bits, mcg);
    return at::mm(x2.to(at::kFloat), w.to(at::kFloat)).to(x.scalar_type());
}

at::Tensor p2b_gemv_batched(const at::Tensor& x, const at::Tensor& trellis_ptrs,
                            const at::Tensor& suh_ptrs, const at::Tensor& svh_ptrs,
                            const at::Tensor& expert_indices, int64_t bits,
                            bool mcg, int64_t mmode) {
    TORCH_CHECK(x.is_cuda(), "batched GEMV requires CUDA tensors");
    return p2b_gemv_batched_cuda(x, trellis_ptrs, suh_ptrs, svh_ptrs,
                                 expert_indices, bits, mcg, mmode);
}

at::Tensor p2b_fused_moe(const at::Tensor& x, at::Tensor& out,
                         const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
                         const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
                         const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
                         const at::Tensor& ids, const at::Tensor& rw,
                         int64_t kg, int64_t ku, int64_t kd, bool mcg,
                         int64_t intermediate_size, float swiglu_limit) {
    return p2b_fused_moe_cuda(x, out, gt, gu, gv, ut, uu, uv, dt, du, dv,
                            ids, rw, kg, ku, kd, mcg, intermediate_size, swiglu_limit);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequant_trellis", &dequant_trellis,
          "Decode an EXL3 trellis tensor into an fp16 weight matrix");
    m.def("exl3_gemv", &exl3_gemv,
          "Native small-m EXL3 GEMV", py::arg("x"), py::arg("trellis"),
          py::arg("suh"), py::arg("svh"), py::arg("K"), py::arg("mcg"),
          py::arg("mmode") = 1);
    m.def("exl3_gemm", &exl3_gemm, "Native tiled EXL3 GEMM",
          py::arg("x"), py::arg("trellis"), py::arg("suh"), py::arg("svh"),
          py::arg("K"), py::arg("mcg"));
    m.def("p2b_gemv_batched", &p2b_gemv_batched,
          "Batched cooperative MoE EXL3 GEMV", py::arg("x"),
          py::arg("trellis_ptrs"), py::arg("suh_ptrs"), py::arg("svh_ptrs"),
          py::arg("expert_indices"), py::arg("K"), py::arg("mcg"),
          py::arg("mmode") = 1);
    m.def("p2b_fused_moe", &p2b_fused_moe,
          "Fused cooperative MoE decode", py::arg("x"), py::arg("out"),
          py::arg("gate_trellis_ptrs"), py::arg("gate_suh_ptrs"), py::arg("gate_svh_ptrs"),
          py::arg("up_trellis_ptrs"), py::arg("up_suh_ptrs"), py::arg("up_svh_ptrs"),
          py::arg("down_trellis_ptrs"), py::arg("down_suh_ptrs"), py::arg("down_svh_ptrs"),
          py::arg("expert_indices"), py::arg("routing_weights"), py::arg("K_gate"),
          py::arg("K_up"), py::arg("K_down"), py::arg("mcg"),
          py::arg("intermediate_size") = 2048, py::arg("swiglu_limit") = 0.0f);
    // Python must not pass TP2 pointers or a clipping request to an older binary.
    m.attr("P2B_MOE_ABI_VERSION") = 2;
    m.def("exl3_fat_gemm", &exl3_fat_gemm, "Native EXL3 fat GEMM for large prefill rows",
          py::arg("a"), py::arg("packed"), py::arg("out"), py::arg("svh"),
          py::arg("K"), py::arg("mcg"), py::arg("mul1"));
    m.def("exl3_fat_gemm_scatter", &exl3_fat_gemm_scatter, "Native EXL3 fat GEMM with token scatter",
          py::arg("a"), py::arg("packed"), py::arg("out"), py::arg("svh"),
          py::arg("token_idx"), py::arg("route_weight"),
          py::arg("K"), py::arg("mcg"), py::arg("mul1"));
}
