#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include "util.h"
#include "util.cuh"

#define EXL3_GEMM_ARGS const half*, const uint16_t*, void*, const int, const int, const int, int*, const half*, half*, const half*
template <int, bool, int, int, int, bool>
__device__ void exl3_gemv_kernel(EXL3_GEMM_ARGS);

// Make the vendor GEMV body device-callable before the fast worklist uses its
// fragment types and decode helpers.
#define P2B_GLOBAL __global__
#define __global__ __device__
#define __launch_bounds__(...)
#include "quant/exl3_gemv_kernel.cuh"
#undef __launch_bounds__
#undef __global__
#define __launch_bounds__(...) __annotate__(launch_bounds(__VA_ARGS__))
#define __global__ __location__(global)

// The standalone QTIP GEMV body is intentionally cooperative and performs one
// full expert at a time.  For the common MoE decode case (m == 1), use the
// same tile math as the fused MoE path but distribute expert/column work over
// the whole cooperative grid.
__device__ __forceinline__ void p2b_run_gemv_tile_2(
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2,
    half* __restrict__ C,
    int kslices,
    int size_k,
    int group,
    int ntiles,
    int warp,
    int lane,
    float (*sh_red)[1][32]) {
    constexpr int WK = 16;
    constexpr int WNT = 2;
    constexpr int PF = 4;
    constexpr int FOLD = 4;
    constexpr int THREADS = 512;
    constexpr int COLS = 32;
    constexpr int TWORDS = 16;

    const int chunk = (kslices + WK - 1) / WK;
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_stride = (size_t)ntiles * TWORDS;
    const half2 hzero = __half2half2(__ushort_as_half(0));
    const bool r0_ok = lane < 4;
    const int i1 = lane >> 1;
    const int x_src_b = i1;
    const int x_src_a = (i1 + 15) & 15;
    const uint32_t* bp = B32 + (size_t)ks0 * slice_stride + group * WNT * TWORDS + lane;

    auto ld_b = [&](int i) -> uint32_t {
        return __ldcs(bp + (size_t)i * slice_stride);
    };

    uint32_t pf[PF];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn) pf[d] = ld_b(d);

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;
            const uint32_t w = pf[d];
            if (i + PF < myn) pf[d] = ld_b(i + PF);

            const size_t a_col = (size_t)(ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_col] : hzero;
            a23[0] = r0_ok ? A2[a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                const int base = (t & 1) << 4;
                const uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                const uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                FragB f0, f1;
                exl3_gemv_ns::dq8_regs_2bits<1>(awv, bwv, lane << 3, f0, f1);
                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
            }

            if ((d + 1) % FOLD == 0 || i + 1 == myn) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
        }
    }

    if (lane < 4) {
        #pragma unroll
        for (int t = 0; t < WNT; ++t)
            #pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int col = t * 16 + f * 8 + (lane & 3) * 2;
                sh_red[warp][0][col + 0] = acc0[t][f].x;
                sh_red[warp][0][col + 1] = acc0[t][f].y;
            }
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < COLS; idx += THREADS) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < WK; ++j) sum += sh_red[j][0][idx];
        C[group * COLS + idx] = __float2half_rn(sum);
    }
    __syncthreads();
}

P2B_GLOBAL __launch_bounds__(512)
void p2b_fast_worklist_kernel(
    const half* A, const int64_t* tptrs, const int64_t* suptrs,
    const int64_t* svptrs, const int32_t* ids, half* C, half* A_had,
    int experts, int size_m, int size_k, int size_n) {
    auto grid = cooperative_groups::this_grid();
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int warps_per_exp = size_k / 128;

    // Each active expert gets its own input Hadamard buffer.
    const int total_had_warps = experts * warps_per_exp;
    const int warp_global = warp + (blockDim.x / 32) * blockIdx.x;
    const int grid_warps = gridDim.x * (blockDim.x / 32);
    for (int widx = warp_global; widx < total_had_warps; widx += grid_warps) {
        const int e = widx / warps_per_exp;
        const int w = widx % warps_per_exp;
        const int src = ids[e];
        const half* su = reinterpret_cast<const half*>(suptrs[src]);
        half* ah = A_had + static_cast<size_t>(e) * size_m * size_k;
        had_hf_r_128_inner<true, false>(
            A + w * 128, ah + w * 128, su + (w * 128) % size_k,
            0.088388347648f);
    }
    grid.sync();

    // Distribute expert/column-group pairs, rather than looping over experts
    // inside every block as the legacy path does.
    const int groups = size_n / 32;
    const int total_work = experts * groups;
    __shared__ float sh_red[16][1][32];
    for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
        const int e = item / groups;
        const int group = item % groups;
        const int src = ids[e];
        const uint32_t* b = reinterpret_cast<const uint32_t*>(tptrs[src]);
        const half2* ah = reinterpret_cast<const half2*>(
            A_had + static_cast<size_t>(e) * size_m * size_k);
        half* c = C + static_cast<size_t>(e) * size_m * size_n;
        p2b_run_gemv_tile_2(b, ah, c, size_k / 16, size_k / 16,
                            group, size_n / 16, warp, lane, sh_red);
    }
    grid.sync();

    // Match standalone GEMV's output Hadamard stage.
    const int out_warps_per_exp = size_n / 128;
    const int total_out_warps = experts * out_warps_per_exp;
    for (int widx = warp_global; widx < total_out_warps; widx += grid_warps) {
        const int e = widx / out_warps_per_exp;
        const int w = widx % out_warps_per_exp;
        const int src = ids[e];
        const half* sv = reinterpret_cast<const half*>(svptrs[src]);
        half* c = C + static_cast<size_t>(e) * size_m * size_n;
        had_hf_r_128_inner<false, true>(
            c + w * 128, c + w * 128, sv + (w * 128) % size_n,
            0.088388347648f);
    }
    grid.sync();
    (void)A;
    (void)size_m;
    (void)total_threads;
}

P2B_GLOBAL __launch_bounds__(256)
void p2b_input_hadamard_kernel(const half* A, const int64_t* suptrs,
                               const int32_t* ids, half* A_had,
                               int experts, int size_k) {
    const int warp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int total = experts * (size_k / 128);
    if (warp >= total) return;
    const int e = warp / (size_k / 128);
    const int w = warp % (size_k / 128);
    const int src = ids[e];
    const half* su = reinterpret_cast<const half*>(suptrs[src]);
    had_hf_r_128_inner<true, false>(
        A + w * 128,
        A_had + static_cast<size_t>(e) * size_k + w * 128,
        su + (w * 128) % size_k,
        0.088388347648f);
}

P2B_GLOBAL __launch_bounds__(256)
void p2b_output_hadamard_kernel(const int64_t* svptrs, const int32_t* ids,
                                half* C, int experts, int size_n) {
    const int warp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int total = experts * (size_n / 128);
    if (warp >= total) return;
    const int e = warp / (size_n / 128);
    const int w = warp % (size_n / 128);
    const int src = ids[e];
    const half* sv = reinterpret_cast<const half*>(svptrs[src]);
    half* c = C + static_cast<size_t>(e) * size_n;
    had_hf_r_128_inner<false, true>(
        c + w * 128, c + w * 128, sv + (w * 128) % size_n,
        0.088388347648f);
}

P2B_GLOBAL __launch_bounds__(512)
void p2b_parallel_worklist_kernel(
    const int64_t* tptrs, const int32_t* ids, half* C, const half* A_had,
    int experts, int size_k, int size_n) {
    const int groups = size_n / 32;
    const int item = blockIdx.x;
    if (item >= experts * groups) return;
    const int e = item / groups;
    const int group = item % groups;
    const int src = ids[e];
    const uint32_t* b = reinterpret_cast<const uint32_t*>(tptrs[src]);
    const half2* ah = reinterpret_cast<const half2*>(
        A_had + static_cast<size_t>(e) * size_k);
    half* c = C + static_cast<size_t>(e) * size_n;
    __shared__ float sh_red[16][1][32];
    p2b_run_gemv_tile_2(b, ah, c, size_k / 16, size_k / 16,
                        group, size_n / 16, threadIdx.x / 32,
                        threadIdx.x % 32, sh_red);
}

template <int BITS, int CB>
P2B_GLOBAL __launch_bounds__(512)
void p2b_worklist_kernel(const half* A, const uint16_t** tptrs,
                        const half** suptrs, const half** svptrs,
                        const int32_t* ids, half* C, half* A_had, int* locks,
                        int experts, int size_m, int size_k, int size_n) {
    constexpr int COLS = 32;
    auto grid = cooperative_groups::this_grid();
    const int group = blockIdx.x;
    const int groups = (size_n + COLS - 1) / COLS;
    for (int e = 0; e < experts; ++e) {
        const int idx = ids[e];
        const half* a = A;
        const uint16_t* b = tptrs[idx];
        void* c = C + static_cast<size_t>(e) * size_m * size_n;
        const half* su = suptrs[idx];
        half* ah = A_had + static_cast<size_t>(e) * size_m * size_k;
        const half* sv = svptrs[idx];
        // The QTIP kernel's grid-stride group loop uses blockIdx.x directly.
        // Parent blocks therefore represent column groups; blocks beyond the
        // active group count simply participate in the required barriers.
        exl3_gemv_kernel<BITS, false, CB, 0, 0, false>(
            a, b, c, size_m, size_k, size_n,
            locks + e * (1 << 20), su, ah, sv);
    }
}

template <int BITS, int CB>
void launch_batched(const at::Tensor& x, const at::Tensor& tp,
                    const at::Tensor& up, const at::Tensor& vp,
                    const at::Tensor& ids, at::Tensor& out, at::Tensor& ah,
                    at::Tensor& locks, int e, int m, int k, int n) {
    int dev = 0; cudaGetDevice(&dev); int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int resident = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
        p2b_worklist_kernel<BITS, CB>, 512, 0);
    const int groups = n / 32;
    const int grid = std::max(1, std::min(e * groups, resident * sms));
    const half* ap = reinterpret_cast<const half*>(x.data_ptr<c10::Half>());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const uint16_t** tp_ptr = reinterpret_cast<const uint16_t**>(tp.data_ptr<int64_t>());
    const half** up_ptr = reinterpret_cast<const half**>(up.data_ptr<int64_t>());
    const half** vp_ptr = reinterpret_cast<const half**>(vp.data_ptr<int64_t>());
    half* out_ptr = reinterpret_cast<half*>(out.data_ptr<c10::Half>());
    half* ah_ptr = reinterpret_cast<half*>(ah.data_ptr<c10::Half>());
    int* lock_ptr = locks.data_ptr<int>();
    int32_t* id_ptr = ids.data_ptr<int32_t>();
    int experts = e, size_m = m, size_k = k, size_n = n;
    void* args[] = {(void*)&ap, (void*)&tp_ptr, (void*)&up_ptr, (void*)&vp_ptr,
                    (void*)&id_ptr, (void*)&out_ptr,
                    (void*)&ah_ptr, (void*)&lock_ptr, (void*)&experts,
                    (void*)&size_m, (void*)&size_k, (void*)&size_n};
    cuda_check(cudaLaunchCooperativeKernel(
        (void*)p2b_worklist_kernel<BITS, CB>, dim3(grid), dim3(512), args, 0, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_batched_fast_2(const at::Tensor& x, const at::Tensor& tp,
                           const at::Tensor& up, const at::Tensor& vp,
                           const at::Tensor& ids, at::Tensor& out,
                           at::Tensor& ah, int e, int m, int k, int n) {
    int dev = 0;
    cudaGetDevice(&dev);
    const int groups = n / 32;
    const half* ap = reinterpret_cast<const half*>(x.data_ptr<c10::Half>());
    const int64_t* tp_ptr = tp.data_ptr<int64_t>();
    const int64_t* up_ptr = up.data_ptr<int64_t>();
    const int64_t* vp_ptr = vp.data_ptr<int64_t>();
    const int32_t* id_ptr = ids.data_ptr<int32_t>();
    half* out_ptr = reinterpret_cast<half*>(out.data_ptr<c10::Half>());
    half* ah_ptr = reinterpret_cast<half*>(ah.data_ptr<c10::Half>());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int had_blocks = (e * (k / 128) + 7) / 8;
    void* had_args[] = {(void*)&ap, (void*)&up_ptr, (void*)&id_ptr,
                        (void*)&ah_ptr, (void*)&e, (void*)&k};
    cuda_check(cudaLaunchKernel((void*)p2b_input_hadamard_kernel,
                                dim3(had_blocks), dim3(256), had_args, 0,
                                stream));

    void* gemv_args[] = {(void*)&tp_ptr, (void*)&id_ptr, (void*)&out_ptr,
                         (void*)&ah_ptr, (void*)&e, (void*)&k, (void*)&n};
    cuda_check(cudaLaunchKernel((void*)p2b_parallel_worklist_kernel,
                                dim3(e * groups), dim3(512), gemv_args, 0,
                                stream));

    const int out_blocks = (e * (n / 128) + 7) / 8;
    void* out_args[] = {(void*)&vp_ptr, (void*)&id_ptr, (void*)&out_ptr,
                        (void*)&e, (void*)&n};
    cuda_check(cudaLaunchKernel((void*)p2b_output_hadamard_kernel,
                                dim3(out_blocks), dim3(256), out_args, 0,
                                stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor p2b_gemv_batched_cuda(const at::Tensor& x,
                                 const at::Tensor& trellis_ptrs,
                                 const at::Tensor& suh_ptrs,
                                 const at::Tensor& svh_ptrs,
                                 const at::Tensor& expert_indices,
                                 int64_t bits, bool mcg, int64_t mmode) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "batched GEMV requires CUDA fp16");
    TORCH_CHECK(trellis_ptrs.is_cuda() && suh_ptrs.is_cuda() && svh_ptrs.is_cuda() && expert_indices.is_cuda(),
                "all batched arguments must be CUDA tensors");
    const int e = static_cast<int>(expert_indices.numel());
    const int m = static_cast<int>(x.numel() / x.size(-1));
    const int k = static_cast<int>(x.size(-1));
    constexpr int n = 2048;
    auto out = at::empty({e, m, n}, x.options().dtype(at::kHalf));
    auto ah = at::empty({e, m, k}, x.options().dtype(at::kHalf));
    auto locks = at::zeros({e * (1 << 20)}, x.options().dtype(at::kInt));
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4, "batched GEMV supports K=2,3,4");
    if (bits == 2 && mcg && m == 1)
        launch_batched_fast_2(x, trellis_ptrs, suh_ptrs, svh_ptrs,
                              expert_indices, out, ah, e, m, k, n);
    else if (bits == 2 && mcg)
        launch_batched<2, 1>(x, trellis_ptrs, suh_ptrs, svh_ptrs, expert_indices, out, ah, locks, e, m, k, n);
    else if (bits == 3 && mcg) launch_batched<3, 1>(x, trellis_ptrs, suh_ptrs, svh_ptrs, expert_indices, out, ah, locks, e, m, k, n);
    else if (bits == 4 && mcg) launch_batched<4, 1>(x, trellis_ptrs, suh_ptrs, svh_ptrs, expert_indices, out, ah, locks, e, m, k, n);
    else TORCH_CHECK(false, "batched GEMV currently requires MCG codebook");
    (void)mmode;
    return out;
}
