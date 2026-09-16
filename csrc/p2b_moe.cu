#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cmath>
#include <limits>

#include "util.h"
#include "util.cuh"
#include "quant/exl3_gemv_kernel.cuh"
// Multi-K additions (ABI 4): exllamav3's dq_dispatch tile decoder
// (exl3_dq.cuh) is templated on bits for K1..8 and is the route-(b)
// decoder for the K5/K6 values the register decoders in exl3_gemv_ns
// (dq8_regs_{2,3,4}bits) never shipped with. The plugin's
// exl3_fat_gemm.cu already compiles this header through the same
// include path (quant/ is on the include list in setup.py).
#include "exl3_dq.cuh"

namespace cg = cooperative_groups;

template <int bits, int cb, int CFG>
__device__ __forceinline__ void run_gemv_tile(
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2,
    half* __restrict__ C,
    int kslices,
    int size_k,
    int group,
    int ntiles,
    int warp,
    int lane,
    float (*sh_red)[1][32])
{
    constexpr int WK = CFG == 0 ? 16 : 8;
    constexpr int WNT = CFG == 0 ? 2 : 4;
    constexpr int PF = CFG == 0 ? 4 : 2;
    constexpr int FOLD = CFG == 0 ? 4 : 2;
    constexpr int THREADS = WK * 32;
    constexpr int COLS = WNT * 16;
    constexpr int TWORDS = 8 * bits;
    constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;
    constexpr int LSTRIDE = bits == 3 ? 24 : 32;

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_stride = (size_t) ntiles * TWORDS;

    const size_t a_row0 = 0;
    const bool r0_ok = lane < 4;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2) {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    } else if constexpr (bits == 3) {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

    auto ld_b = [&] (int i, int l) -> uint32_t {
        if constexpr (bits == 3)
            return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else
            return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
    };

    uint32_t pf[PF][LOADS];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn)
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = ld_b(d, l);

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }

            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                FragB f0, f1;
                if constexpr (bits == 4) {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                } else if constexpr (bits == 2) {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                } else {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }

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

    // Warp reduction
    if (lane < 4) {
        #pragma unroll
        for (int t = 0; t < WNT; ++t) {
            #pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int col = t * 16 + f * 8 + (lane & 3) * 2;
                sh_red[warp][0][col + 0] = acc0[t][f].x;
                sh_red[warp][0][col + 1] = acc0[t][f].y;
            }
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < COLS; idx += THREADS) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < WK; ++j)
            sum += sh_red[j][0][idx];
        const int col = group * COLS + idx;
        C[col] = __float2half_rn(sum);
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Multi-K additions (ABI 4). Everything above is the uniform-K path and stays
// byte-identical to the staged original; the blocks below only add to it.
// ---------------------------------------------------------------------------

// Route-(b) tile function: same contract and calling convention as
// run_gemv_tile above (single-row A broadcast through half2 fragments,
// mma_ab_h tensor-core path, fp16 fold every FOLD slices, sh_red warp
// reduction, one 32-column group writeback), but each 16x16 B tile is
// decoded through exllamav3's dq_dispatch (exl3_dq.cuh), which is templated
// on bits for K1..8, instead of the K2/3/4-only register decoders. Tile
// addressing is the standard EXL3 trellis layout [rows/16, cols/16, 16*K]
// int16: one 16x16 tile is 8*bits uint32 words at
//   B32 + row_tile * (ntiles * TWORDS) + col_tile * TWORDS
// exactly the indexing run_gemv_tile derives for its register loads.
template <int bits, int cb, int CFG>
__device__ __forceinline__ void run_gemm_tile_dq(
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2,
    half* __restrict__ C,
    int kslices,
    int size_k,
    int group,
    int ntiles,
    int warp,
    int lane,
    float (*sh_red)[1][32])
{
    constexpr int WK = CFG == 0 ? 16 : 8;
    constexpr int WNT = CFG == 0 ? 2 : 4;
    constexpr int PF = CFG == 0 ? 4 : 2;
    constexpr int FOLD = CFG == 0 ? 4 : 2;
    constexpr int THREADS = WK * 32;
    constexpr int COLS = WNT * 16;
    constexpr int TWORDS = 8 * bits;

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_stride = (size_t) ntiles * TWORDS;

    const bool r0_ok = lane < 4;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;

            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_col] : hzero;
            a23[0] = r0_ok ? A2[a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                const uint32_t* tw = B32
                    + (size_t) (ks0 + i) * slice_stride
                    + (size_t) (group * WNT + t) * TWORDS;
                FragB f0, f1;
                dq_dispatch<bits, cb>(tw, lane << 3, f0, f1);
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

    // Warp reduction
    if (lane < 4) {
        #pragma unroll
        for (int t = 0; t < WNT; ++t) {
            #pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int col = t * 16 + f * 8 + (lane & 3) * 2;
                sh_red[warp][0][col + 0] = acc0[t][f].x;
                sh_red[warp][0][col + 1] = acc0[t][f].y;
            }
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < COLS; idx += THREADS) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < WK; ++j)
            sum += sh_red[j][0][idx];
        const int col = group * COLS + idx;
        C[col] = __float2half_rn(sum);
    }
    __syncthreads();
}

// Runtime-K dispatcher: each work item reads its expert's K from an int8
// table and switches over the compiled tile instantiations. The
// instantiation count is the number of distinct K values per projection
// (not the K cross-product): K2/3/4 use the register decoders, K5/6 the
// dq_dispatch tile function above. The switch operand is block-uniform
// (src = ids[e] is uniform per work item), so no warp divergence is added.
template <int CB>
__device__ __forceinline__ void run_gemv_tile_k(
    int bits,
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2,
    half* __restrict__ C,
    int kslices,
    int size_k,
    int group,
    int ntiles,
    int warp,
    int lane,
    float (*sh_red)[1][32])
{
    switch (bits) {
        case 2: run_gemv_tile<2, CB, 0>(B32, A2, C, kslices, size_k, group, ntiles, warp, lane, sh_red); break;
        case 3: run_gemv_tile<3, CB, 0>(B32, A2, C, kslices, size_k, group, ntiles, warp, lane, sh_red); break;
        case 4: run_gemv_tile<4, CB, 0>(B32, A2, C, kslices, size_k, group, ntiles, warp, lane, sh_red); break;
        case 5: run_gemm_tile_dq<5, CB, 0>(B32, A2, C, kslices, size_k, group, ntiles, warp, lane, sh_red); break;
        case 6: run_gemm_tile_dq<6, CB, 0>(B32, A2, C, kslices, size_k, group, ntiles, warp, lane, sh_red); break;
        default: {
            // Unreachable: Python validates the table values (2..6) when the
            // tables are built in finalize. Zero the tile deterministically
            // so a bad table can never propagate garbage into the reduction.
            for (int idx = threadIdx.x; idx < 32; idx += 512)
                C[group * 32 + idx] = __float2half_rn(0.0f);
            break;
        }
    }
}

template <int BITS>
__global__ __launch_bounds__(512)
void p2b_moe_batched_kernel(
    const half* __restrict__ x,
    const int64_t* __restrict__ gt_ptrs,
    const int64_t* __restrict__ gu_ptrs,
    const int64_t* __restrict__ gv_ptrs,
    const int64_t* __restrict__ ut_ptrs,
    const int64_t* __restrict__ uu_ptrs,
    const int64_t* __restrict__ uv_ptrs,
    const int64_t* __restrict__ dt_ptrs,
    const int64_t* __restrict__ du_ptrs,
    const int64_t* __restrict__ dv_ptrs,
    const int32_t* __restrict__ ids,
    const half* __restrict__ rw,
    half* __restrict__ gate,
    half* __restrict__ up,
    half* __restrict__ down,
    half* __restrict__ out,
    half* __restrict__ had_gate,
    half* __restrict__ had_up,
    half* __restrict__ had_down,
    float* __restrict__ accum,
    int experts,
    int m,
    int hidden,
    int inter,
    float swiglu_limit)
{
    auto grid = cg::this_grid();
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    const int ntiles_gate = inter / 16;
    const int kslices_gate = hidden / 16;
    const int num_groups_gate = inter / 32;

    const int ntiles_down = hidden / 16;
    const int kslices_down = inter / 16;
    const int num_groups_down = hidden / 32;

    __shared__ float sh_red[16][1][32];

    // Zero accum
    for (int j = tid; j < m * hidden; j += total_threads)
        accum[j] = 0.0f;

    // Phase 1: Input Hadamard for Gate and Up across all active experts
    {
        int warps_per_exp = hidden / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[e];
            const half* gu_e = reinterpret_cast<const half*>(gu_ptrs[src]);
            const half* uu_e = reinterpret_cast<const half*>(uu_ptrs[src]);
            half* hg_e = had_gate + e * hidden;
            half* hu_e = had_up + e * hidden;

            had_hf_r_128_inner<true, false>(x + w * 128, hg_e + w * 128, gu_e + (w * 128) % hidden, 0.088388347648f);
            had_hf_r_128_inner<true, false>(x + w * 128, hu_e + w * 128, uu_e + (w * 128) % hidden, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 2: Batched Gate & Up GEMV across all active experts
    {
        int total_work = 2 * experts * num_groups_gate;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int is_up = item & 1;
            int rem = item >> 1;
            int e = rem / num_groups_gate;
            int group = rem % num_groups_gate;
            int src = ids[e];

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(is_up ? ut_ptrs[src] : gt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>((is_up ? had_up : had_gate) + e * hidden);
            half* C = (is_up ? up : gate) + e * inter;

            run_gemv_tile<BITS, 1, 0>(B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);
        }
        grid.sync();
    }

    // Epilogue Hadamard on Gate and Up
    {
        int warps_per_exp = inter / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[e];
            const half* gv_e = reinterpret_cast<const half*>(gv_ptrs[src]);
            const half* uv_e = reinterpret_cast<const half*>(uv_ptrs[src]);
            half* gp_e = gate + e * inter;
            half* up_e = up + e * inter;

            had_hf_r_128_inner<false, true>(gp_e + w * 128, gp_e + w * 128, gv_e + (w * 128) % inter, 0.088388347648f);
            had_hf_r_128_inner<false, true>(up_e + w * 128, up_e + w * 128, uv_e + (w * 128) % inter, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 3: SwiGLU activation + Down input Hadamard across all active experts
    {
        // Match vLLM's input-clipped SwiGLU. Zero preserves the plain activation.
        int total_elements = experts * inter;
        for (int j = tid; j < total_elements; j += total_threads) {
            float g = __half2float(gate[j]);
            float u = __half2float(up[j]);
            if (swiglu_limit > 0.0f) {
                g = fminf(g, swiglu_limit);
                u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
            }
            float s = g / (1.0f + expf(-g));
            had_down[j] = __float2half(s * u);
        }
        grid.sync();

        // Down input Hadamard on had_down
        int warps_per_exp = inter / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[e];
            const half* du_e = reinterpret_cast<const half*>(du_ptrs[src]);
            half* hd_e = had_down + e * inter;

            had_hf_r_128_inner<true, false>(hd_e + w * 128, hd_e + w * 128, du_e + (w * 128) % inter, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 4: Batched Down GEMV across all active experts
    {
        int total_work = experts * num_groups_down;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int e = item / num_groups_down;
            int group = item % num_groups_down;
            int src = ids[e];

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(dt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>(had_down + e * inter);
            half* C = down + e * hidden;

            run_gemv_tile<BITS, 1, 0>(B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);
        }
        grid.sync();
    }

    // Down output Hadamard and atomic accumulation into accum
    {
        int warps_per_exp = hidden / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[e];
            const half* dv_e = reinterpret_cast<const half*>(dv_ptrs[src]);
            half* dp_e = down + e * hidden;

            had_hf_r_128_inner<false, true>(dp_e + w * 128, dp_e + w * 128, dv_e + (w * 128) % hidden, 0.088388347648f);
        }
        grid.sync();

        // Weighted reduction into accum
        int total_elements = experts * hidden;
        for (int j = tid; j < total_elements; j += total_threads) {
            int e = j / hidden;
            int col = j % hidden;
            float w = __half2float(rw[e]);
            atomicAdd(accum + col, w * __half2float(down[j]));
        }
        grid.sync();
    }

    // Write back to out
    for (int j = tid; j < m * hidden; j += total_threads) {
        out[j] = __float2half(accum[j]);
    }
}

// Mixed-K cooperative MoE kernel (ABI 4): ONE launch for a routing list whose
// experts carry per-expert K (SAGE-allocated packs). The phase structure is
// identical to p2b_moe_batched_kernel above; the differences are:
//   * per-expert K comes from int8 tables kg_tab/ku_tab/kd_tab indexed by the
//     block-uniform src = ids[e], and phases 2/4 dispatch through
//     run_gemv_tile_k instead of the compile-time BITS template;
//   * n_local bounds the pointer tables: routing slots with
//     src >= n_local (the EP sentinel produced by map_topk_to_local) are
//     skipped in every phase, so a non-local slot neither streams its
//     expert's weights nor lets uninitialized fp16 scratch (possibly NaN)
//     reach the fp32 accumulation. The guard in the weighted reduction is
//     load-bearing: an uninitialized half can be NaN and 0 * NaN = NaN would
//     poison the output even with a zeroed routing weight.
__global__ __launch_bounds__(512)
void p2b_moe_mixedk_kernel(
    const half* __restrict__ x,
    const int64_t* __restrict__ gt_ptrs,
    const int64_t* __restrict__ gu_ptrs,
    const int64_t* __restrict__ gv_ptrs,
    const int64_t* __restrict__ ut_ptrs,
    const int64_t* __restrict__ uu_ptrs,
    const int64_t* __restrict__ uv_ptrs,
    const int64_t* __restrict__ dt_ptrs,
    const int64_t* __restrict__ du_ptrs,
    const int64_t* __restrict__ dv_ptrs,
    const int32_t* __restrict__ ids,
    const half* __restrict__ rw,
    const int8_t* __restrict__ kg_tab,
    const int8_t* __restrict__ ku_tab,
    const int8_t* __restrict__ kd_tab,
    int n_local,
    half* __restrict__ gate,
    half* __restrict__ up,
    half* __restrict__ down,
    half* __restrict__ out,
    half* __restrict__ had_gate,
    half* __restrict__ had_up,
    half* __restrict__ had_down,
    float* __restrict__ accum,
    int experts,
    int m,
    int hidden,
    int inter,
    float swiglu_limit)
{
    auto grid = cg::this_grid();
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    const int ntiles_gate = inter / 16;
    const int kslices_gate = hidden / 16;
    const int num_groups_gate = inter / 32;

    const int ntiles_down = hidden / 16;
    const int kslices_down = inter / 16;
    const int num_groups_down = hidden / 32;

    __shared__ float sh_red[16][1][32];

    // Zero accum
    for (int j = tid; j < m * hidden; j += total_threads)
        accum[j] = 0.0f;

    // Phase 1: Input Hadamard for Gate and Up across all active experts
    {
        int warps_per_exp = hidden / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[e];
            if (src < 0 || src >= n_local) continue;
            const half* gu_e = reinterpret_cast<const half*>(gu_ptrs[src]);
            const half* uu_e = reinterpret_cast<const half*>(uu_ptrs[src]);
            half* hg_e = had_gate + e * hidden;
            half* hu_e = had_up + e * hidden;

            had_hf_r_128_inner<true, false>(x + w * 128, hg_e + w * 128, gu_e + (w * 128) % hidden, 0.088388347648f);
            had_hf_r_128_inner<true, false>(x + w * 128, hu_e + w * 128, uu_e + (w * 128) % hidden, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 2: Batched Gate & Up GEMV across all active experts
    {
        int total_work = 2 * experts * num_groups_gate;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int is_up = item & 1;
            int rem = item >> 1;
            int e = rem / num_groups_gate;
            int group = rem % num_groups_gate;
            int src = ids[e];
            if (src < 0 || src >= n_local) continue;

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(is_up ? ut_ptrs[src] : gt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>((is_up ? had_up : had_gate) + e * hidden);
            half* C = (is_up ? up : gate) + e * inter;
            const int kb = (int) (is_up ? ku_tab[src] : kg_tab[src]);

            run_gemv_tile_k<1>(kb, B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);
        }
        grid.sync();
    }

    // Epilogue Hadamard on Gate and Up
    {
        int warps_per_exp = inter / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[e];
            if (src < 0 || src >= n_local) continue;
            const half* gv_e = reinterpret_cast<const half*>(gv_ptrs[src]);
            const half* uv_e = reinterpret_cast<const half*>(uv_ptrs[src]);
            half* gp_e = gate + e * inter;
            half* up_e = up + e * inter;

            had_hf_r_128_inner<false, true>(gp_e + w * 128, gp_e + w * 128, gv_e + (w * 128) % inter, 0.088388347648f);
            had_hf_r_128_inner<false, true>(up_e + w * 128, up_e + w * 128, uv_e + (w * 128) % inter, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 3: SwiGLU activation + Down input Hadamard across all active experts
    {
        // Match vLLM's input-clipped SwiGLU. Zero preserves the plain activation.
        // Sentinel slots are skipped so uninitialized gate/up rows (possibly
        // NaN/Inf) never enter the activation pipeline at all.
        int total_elements = experts * inter;
        for (int j = tid; j < total_elements; j += total_threads) {
            int e3 = j / inter;
            int src3 = ids[e3];
            if (src3 < 0 || src3 >= n_local) continue;
            float g = __half2float(gate[j]);
            float u = __half2float(up[j]);
            if (swiglu_limit > 0.0f) {
                g = fminf(g, swiglu_limit);
                u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
            }
            float s = g / (1.0f + expf(-g));
            had_down[j] = __float2half(s * u);
        }
        grid.sync();

        // Down input Hadamard on had_down
        int warps_per_exp = inter / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[e];
            if (src < 0 || src >= n_local) continue;
            const half* du_e = reinterpret_cast<const half*>(du_ptrs[src]);
            half* hd_e = had_down + e * inter;

            had_hf_r_128_inner<true, false>(hd_e + w * 128, hd_e + w * 128, du_e + (w * 128) % inter, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 4: Batched Down GEMV across all active experts
    {
        int total_work = experts * num_groups_down;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int e = item / num_groups_down;
            int group = item % num_groups_down;
            int src = ids[e];
            if (src < 0 || src >= n_local) continue;

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(dt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>(had_down + e * inter);
            half* C = down + e * hidden;
            const int kb = (int) kd_tab[src];

            run_gemv_tile_k<1>(kb, B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);
        }
        grid.sync();
    }

    // Down output Hadamard and atomic accumulation into accum
    {
        int warps_per_exp = hidden / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[e];
            if (src < 0 || src >= n_local) continue;
            const half* dv_e = reinterpret_cast<const half*>(dv_ptrs[src]);
            half* dp_e = down + e * hidden;

            had_hf_r_128_inner<false, true>(dp_e + w * 128, dp_e + w * 128, dv_e + (w * 128) % hidden, 0.088388347648f);
        }
        grid.sync();

        // Weighted reduction into accum. The sentinel guard is load-bearing:
        // skipped experts leave `down` uninitialized and an uninitialized
        // half can be NaN, so 0 * NaN = NaN would poison accum even though
        // the routing weight is zero.
        int total_elements = experts * hidden;
        for (int j = tid; j < total_elements; j += total_threads) {
            int e = j / hidden;
            int col = j % hidden;
            int src5 = ids[e];
            if (src5 < 0 || src5 >= n_local) continue;
            float w = __half2float(rw[e]);
            atomicAdd(accum + col, w * __half2float(down[j]));
        }
        grid.sync();
    }

    // Write back to out
    for (int j = tid; j < m * hidden; j += total_threads) {
        out[j] = __float2half(accum[j]);
    }
}

template <int BITS>
static void launch_moe_batched(
    const at::Tensor& x, const at::Tensor& gt, const at::Tensor& gu,
    const at::Tensor& gv, const at::Tensor& ut, const at::Tensor& uu,
    const at::Tensor& uv, const at::Tensor& dt, const at::Tensor& du,
    const at::Tensor& dv, const at::Tensor& ids, const at::Tensor& rw,
    at::Tensor& out, at::Tensor& gate, at::Tensor& up, at::Tensor& down,
    at::Tensor& had_gate, at::Tensor& had_up, at::Tensor& had_down,
    at::Tensor& accum, int e, int m, int hidden, int inter, float swiglu_limit)
{
    int dev = 0, sms = 0, resident = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    void* kernel = (void*) p2b_moe_batched_kernel<BITS>;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 512, 0);
    const int grid = std::max(1, resident * sms);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const half* xp = reinterpret_cast<const half*>(x.data_ptr<c10::Half>());
    const int64_t* gtp = gt.data_ptr<int64_t>();
    const int64_t* gup = gu.data_ptr<int64_t>();
    const int64_t* gvp = gv.data_ptr<int64_t>();
    const int64_t* utp = ut.data_ptr<int64_t>();
    const int64_t* uup = uu.data_ptr<int64_t>();
    const int64_t* uvp = uv.data_ptr<int64_t>();
    const int64_t* dtp = dt.data_ptr<int64_t>();
    const int64_t* dup = du.data_ptr<int64_t>();
    const int64_t* dvp = dv.data_ptr<int64_t>();
    const int32_t* idp = ids.data_ptr<int32_t>();
    const half* rwp = reinterpret_cast<const half*>(rw.data_ptr<c10::Half>());

    half* gp = reinterpret_cast<half*>(gate.data_ptr<c10::Half>());
    half* up_p = reinterpret_cast<half*>(up.data_ptr<c10::Half>());
    half* dp = reinterpret_cast<half*>(down.data_ptr<c10::Half>());
    half* op = reinterpret_cast<half*>(out.data_ptr<c10::Half>());
    half* hg_p = reinterpret_cast<half*>(had_gate.data_ptr<c10::Half>());
    half* hu_p = reinterpret_cast<half*>(had_up.data_ptr<c10::Half>());
    half* hd_p = reinterpret_cast<half*>(had_down.data_ptr<c10::Half>());
    float* accp = accum.data_ptr<float>();

    void* args[] = {
        (void*)&xp, (void*)&gtp, (void*)&gup, (void*)&gvp,
        (void*)&utp, (void*)&uup, (void*)&uvp,
        (void*)&dtp, (void*)&dup, (void*)&dvp,
        (void*)&idp, (void*)&rwp,
        (void*)&gp, (void*)&up_p, (void*)&dp, (void*)&op,
        (void*)&hg_p, (void*)&hu_p, (void*)&hd_p, (void*)&accp,
        (void*)&e, (void*)&m, (void*)&hidden, (void*)&inter, (void*)&swiglu_limit
    };

    cuda_check(cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(512), args, 0, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void launch_moe_mixedk(
    const at::Tensor& x, const at::Tensor& gt, const at::Tensor& gu,
    const at::Tensor& gv, const at::Tensor& ut, const at::Tensor& uu,
    const at::Tensor& uv, const at::Tensor& dt, const at::Tensor& du,
    const at::Tensor& dv, const at::Tensor& ids, const at::Tensor& rw,
    const at::Tensor& kg_tab, const at::Tensor& ku_tab, const at::Tensor& kd_tab,
    int n_local,
    at::Tensor& out, at::Tensor& gate, at::Tensor& up, at::Tensor& down,
    at::Tensor& had_gate, at::Tensor& had_up, at::Tensor& had_down,
    at::Tensor& accum, int e, int m, int hidden, int inter, float swiglu_limit)
{
    // Re-query occupancy for the (larger) mixed-K kernel; never reuse the
    // uniform-K template's resident count.
    int dev = 0, sms = 0, resident = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    void* kernel = (void*) p2b_moe_mixedk_kernel;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 512, 0);
    const int grid = std::max(1, resident * sms);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const half* xp = reinterpret_cast<const half*>(x.data_ptr<c10::Half>());
    const int64_t* gtp = gt.data_ptr<int64_t>();
    const int64_t* gup = gu.data_ptr<int64_t>();
    const int64_t* gvp = gv.data_ptr<int64_t>();
    const int64_t* utp = ut.data_ptr<int64_t>();
    const int64_t* uup = uu.data_ptr<int64_t>();
    const int64_t* uvp = uv.data_ptr<int64_t>();
    const int64_t* dtp = dt.data_ptr<int64_t>();
    const int64_t* dup = du.data_ptr<int64_t>();
    const int64_t* dvp = dv.data_ptr<int64_t>();
    const int32_t* idp = ids.data_ptr<int32_t>();
    const half* rwp = reinterpret_cast<const half*>(rw.data_ptr<c10::Half>());
    const int8_t* kgp = kg_tab.data_ptr<int8_t>();
    const int8_t* kup = ku_tab.data_ptr<int8_t>();
    const int8_t* kdp = kd_tab.data_ptr<int8_t>();

    half* gp = reinterpret_cast<half*>(gate.data_ptr<c10::Half>());
    half* up_p = reinterpret_cast<half*>(up.data_ptr<c10::Half>());
    half* dp = reinterpret_cast<half*>(down.data_ptr<c10::Half>());
    half* op = reinterpret_cast<half*>(out.data_ptr<c10::Half>());
    half* hg_p = reinterpret_cast<half*>(had_gate.data_ptr<c10::Half>());
    half* hu_p = reinterpret_cast<half*>(had_up.data_ptr<c10::Half>());
    half* hd_p = reinterpret_cast<half*>(had_down.data_ptr<c10::Half>());
    float* accp = accum.data_ptr<float>();

    void* args[] = {
        (void*)&xp, (void*)&gtp, (void*)&gup, (void*)&gvp,
        (void*)&utp, (void*)&uup, (void*)&uvp,
        (void*)&dtp, (void*)&dup, (void*)&dvp,
        (void*)&idp, (void*)&rwp,
        (void*)&kgp, (void*)&kup, (void*)&kdp, (void*)&n_local,
        (void*)&gp, (void*)&up_p, (void*)&dp, (void*)&op,
        (void*)&hg_p, (void*)&hu_p, (void*)&hd_p, (void*)&accp,
        (void*)&e, (void*)&m, (void*)&hidden, (void*)&inter, (void*)&swiglu_limit
    };

    cuda_check(cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(512), args, 0, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Host entry for the mixed-K cooperative MoE kernel (ABI 4): one launch for a
// per-expert-K routing list. Same contract as p2b_fused_moe_cuda above
// (single input row, 128-aligned geometry, pointer tables validated by the
// Python caller) except:
//   * the K triple is per expert, carried by int8 device tables
//     kg_tab/ku_tab/kd_tab with one entry per pointer-table slot
//     (K = trellis.shape[-1] / 16 per projection, built by Python finalize);
//   * ids may carry the n_local sentinel (== pointer-table length) for
//     non-local routed slots; the kernel skips those slots in every phase,
//     so their routing weights are irrelevant (Python zeroes them for
//     cleanliness only).
// Table VALUES are validated on the host at load time (Python finalize);
// this entry point must not add a device->host read on the decode path.
at::Tensor p2b_fused_moe_mk_cuda(const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rw,
    const at::Tensor& kg_tab, const at::Tensor& ku_tab, const at::Tensor& kd_tab,
    int64_t n_local, bool mcg, int64_t intermediate_size, float swiglu_limit) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "mixed-K fused MoE requires CUDA fp16 input");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "mixed-K fused MoE output must be CUDA fp16");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
                "mixed-K fused MoE requires exactly one input row");
    TORCH_CHECK(out.sizes() == x.sizes(), "mixed-K fused MoE output shape must match input");
    TORCH_CHECK(x.size(1) > 0 && x.size(1) % 128 == 0,
                "mixed-K fused MoE hidden width must be a positive multiple of 128");
    TORCH_CHECK(intermediate_size > 0 && intermediate_size % 128 == 0,
                "mixed-K fused MoE local intermediate width must be a positive multiple of 128");
    TORCH_CHECK(x.size(1) <= std::numeric_limits<int>::max() &&
                intermediate_size <= std::numeric_limits<int>::max(),
                "mixed-K fused MoE dimensions exceed int32 kernel indexing");
    TORCH_CHECK(std::isfinite(swiglu_limit) && swiglu_limit >= 0.0f,
                "mixed-K fused MoE SwiGLU limit must be finite and nonnegative (0 disables clipping)");
    TORCH_CHECK(mcg, "mixed-K fused MoE currently instantiates the MCG codebook only (cb = 1)");
    TORCH_CHECK(ids.dim() == 1 && ids.scalar_type() == at::kInt && ids.numel() > 0,
                "mixed-K fused MoE expert indices must be a nonempty int32 routing vector");
    TORCH_CHECK(rw.scalar_type() == at::kHalf && rw.numel() == ids.numel(),
                "mixed-K fused MoE requires one fp16 routing weight per expert index");
    const at::Tensor* tensors[] = {&x, &out, &ids, &rw, &gt, &gu, &gv, &ut, &uu, &uv, &dt, &du, &dv,
                                   &kg_tab, &ku_tab, &kd_tab};
    for (const auto* tensor : tensors) {
        TORCH_CHECK(tensor->device() == x.device() && tensor->is_contiguous(),
                    "mixed-K fused MoE tensors must be contiguous and on the input CUDA device");
    }
    for (const auto* ptrs : {&gt, &gu, &gv, &ut, &uu, &uv, &dt, &du, &dv}) {
        TORCH_CHECK(ptrs->dim() == 1 && ptrs->scalar_type() == at::kLong &&
                    ptrs->numel() == gt.numel() && ptrs->numel() > 0,
                    "mixed-K fused MoE pointer tables must be equally sized nonempty int64 vectors");
    }
    for (const auto* tab : {&kg_tab, &ku_tab, &kd_tab}) {
        TORCH_CHECK(tab->is_cuda() && tab->dim() == 1 && tab->is_contiguous() &&
                    tab->scalar_type() == at::kChar && tab->numel() == gt.numel(),
                    "mixed-K fused MoE requires int8 CUDA K tables with one entry per pointer-table slot");
    }
    TORCH_CHECK(n_local >= 1 && n_local <= static_cast<int64_t>(gt.numel()),
                "mixed-K fused MoE n_local must bound the pointer tables (1 <= n_local <= table length)");
    const c10::cuda::CUDAGuard device_guard(x.device());
    const int e = static_cast<int>(ids.numel());
    constexpr int m = 1;
    const int hidden = static_cast<int>(x.size(1));
    const int inter = static_cast<int>(intermediate_size);

    auto gate = at::empty({e, m, inter}, x.options());
    auto up = at::empty({e, m, inter}, x.options());
    auto down = at::empty({e, m, hidden}, x.options());
    auto had_gate = at::empty({e, m, hidden}, x.options());
    auto had_up = at::empty({e, m, hidden}, x.options());
    auto had_down = at::empty({e, m, inter}, x.options());
    auto accum = at::zeros({m, hidden}, x.options().dtype(at::kFloat));

    launch_moe_mixedk(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw,
                      kg_tab, ku_tab, kd_tab, static_cast<int>(n_local),
                      out, gate, up, down, had_gate, had_up, had_down,
                      accum, e, m, hidden, inter, swiglu_limit);

    return out;
}

at::Tensor p2b_fused_moe_cuda(const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rw, int64_t kg, int64_t ku,
    int64_t kd, bool mcg, int64_t intermediate_size, float swiglu_limit) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "fused MoE requires CUDA fp16 input");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "fused MoE output must be CUDA fp16");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
                "fused MoE requires exactly one input row");
    TORCH_CHECK(out.sizes() == x.sizes(), "fused MoE output shape must match input");
    TORCH_CHECK(x.size(1) > 0 && x.size(1) % 128 == 0,
                "fused MoE hidden width must be a positive multiple of 128");
    TORCH_CHECK(intermediate_size > 0 && intermediate_size % 128 == 0,
                "fused MoE local intermediate width must be a positive multiple of 128");
    TORCH_CHECK(x.size(1) <= std::numeric_limits<int>::max() &&
                intermediate_size <= std::numeric_limits<int>::max(),
                "fused MoE dimensions exceed int32 kernel indexing");
    TORCH_CHECK(std::isfinite(swiglu_limit) && swiglu_limit >= 0.0f,
                "fused MoE SwiGLU limit must be finite and nonnegative (0 disables clipping)");
    TORCH_CHECK(mcg && kg == ku && ku == kd && (kg == 2 || kg == 3 || kg == 4), "unsupported fused MoE K");
    TORCH_CHECK(ids.dim() == 1 && ids.scalar_type() == at::kInt && ids.numel() > 0,
                "fused MoE expert indices must be a nonempty int32 routing vector");
    TORCH_CHECK(rw.scalar_type() == at::kHalf && rw.numel() == ids.numel(),
                "fused MoE requires one fp16 routing weight per expert index");
    // Pointer tables describe already-loaded tensors. Their pointee shapes and
    // expert IDs are validated/prepared by the Python caller, without a host sync.
    const at::Tensor* tensors[] = {&x, &out, &ids, &rw, &gt, &gu, &gv, &ut, &uu, &uv, &dt, &du, &dv};
    for (const auto* tensor : tensors) {
        TORCH_CHECK(tensor->device() == x.device() && tensor->is_contiguous(),
                    "fused MoE tensors must be contiguous and on the input CUDA device");
    }
    for (const auto* ptrs : {&gt, &gu, &gv, &ut, &uu, &uv, &dt, &du, &dv}) {
        TORCH_CHECK(ptrs->dim() == 1 && ptrs->scalar_type() == at::kLong &&
                    ptrs->numel() == gt.numel() && ptrs->numel() > 0,
                    "fused MoE pointer tables must be equally sized nonempty int64 vectors");
    }
    const c10::cuda::CUDAGuard device_guard(x.device());
    const int e = static_cast<int>(ids.numel());
    constexpr int m = 1;
    const int hidden = static_cast<int>(x.size(1));
    const int inter = static_cast<int>(intermediate_size);

    auto gate = at::empty({e, m, inter}, x.options());
    auto up = at::empty({e, m, inter}, x.options());
    auto down = at::empty({e, m, hidden}, x.options());
    auto had_gate = at::empty({e, m, hidden}, x.options());
    auto had_up = at::empty({e, m, hidden}, x.options());
    auto had_down = at::empty({e, m, inter}, x.options());
    auto accum = at::zeros({m, hidden}, x.options().dtype(at::kFloat));

    if (kg == 2) launch_moe_batched<2>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, swiglu_limit);
    else if (kg == 3) launch_moe_batched<3>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, swiglu_limit);
    else if (kg == 4) launch_moe_batched<4>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, swiglu_limit);

    return out;
}