#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <mutex>
#include "heterogeneous_moe.h"
#include "native/quant/exl3_moe_kernel.cuh"

// The adapter generates descriptors from verified host metadata, and keeps all
// pointed-to tensors leased through a completion event. This internal entry
// point validates tensor geometry without downloading metadata on every launch.
void sage_heterogeneous_moe(
    const at::Tensor& x, const at::Tensor& out, const at::Tensor& counts,
    const at::Tensor& tokens, const at::Tensor& weights,
    const at::Tensor& temp_g, const at::Tensor& temp_u,
    const at::Tensor& temp_ig, const at::Tensor& temp_iu,
    const at::Tensor& bits, const at::Tensor& pointers,
    const at::Tensor& locks, double limit) {
    TORCH_CHECK(x.is_cuda(), "CUDA activation required");
    const c10::cuda::CUDAGuard guard(x.device());
    for (const auto* tensor : {&x,&out,&counts,&tokens,&weights,&temp_g,&temp_u,&temp_ig,&temp_iu,&bits,&pointers,&locks}) {
        TORCH_CHECK(tensor->device()==x.device() && tensor->is_contiguous(), "Same-device contiguous tensors required");
    }
    TORCH_CHECK(x.scalar_type()==at::kHalf && x.dim()==2 && x.size(0)>0 && x.size(1)==5120, "Exact activation geometry required");
    TORCH_CHECK(out.scalar_type()==at::kFloat && out.sizes()==x.sizes(), "FP32 output geometry required");
    TORCH_CHECK(counts.scalar_type()==at::kLong && counts.dim()==1 && counts.size(0)>=2 && counts.size(0)<=17, "At most sixteen leased experts");
    const int experts=counts.size(0)-1;
    TORCH_CHECK(tokens.scalar_type()==at::kLong && tokens.dim()==1 && tokens.size(0)%x.size(0)==0, "Integral route geometry required");
    TORCH_CHECK(weights.scalar_type()==at::kHalf && weights.sizes()==tokens.sizes(), "FP16 sorted routing weights required");
    TORCH_CHECK(bits.scalar_type()==at::kLong && bits.dim()==2 && bits.size(0)==experts && bits.size(1)==3, "One K triplet per expert required");
    TORCH_CHECK(pointers.scalar_type()==at::kLong && pointers.dim()==2 && pointers.size(0)==9 && pointers.size(1)==experts, "Nine pointer tables required");
    TORCH_CHECK(temp_g.scalar_type()==at::kHalf && temp_u.scalar_type()==at::kHalf && temp_g.dim()==3 && temp_g.sizes()==temp_u.sizes(), "Gate/up scratch geometry required");
    TORCH_CHECK(temp_ig.scalar_type()==at::kHalf && temp_iu.scalar_type()==at::kHalf && temp_ig.dim()==3 && temp_ig.sizes()==temp_iu.sizes(), "Intermediate scratch geometry required");
    TORCH_CHECK(temp_g.size(2)==5120 && temp_ig.size(2)==2304 && temp_g.size(0)==temp_ig.size(0) && temp_g.size(1)==temp_ig.size(1), "Exact checkpoint scratch dimensions required");
    TORCH_CHECK(temp_g.size(1)==16, "Only thin expert routes are qualified");
    TORCH_CHECK(locks.scalar_type()==at::kInt && locks.dim()==1 && locks.numel()==MAX_TILES_C+2*MAX_BARRIERS+MOE_SCHED_INTS, "Dedicated initialized lock buffer required");
    TORCH_CHECK(std::isfinite(limit) && limit>=0, "Finite activation limit required");
    const auto* properties=at::cuda::getDeviceProperties(x.get_device());
    TORCH_CHECK(properties->major==12 && properties->minor==1, "SM121 prototype only");
    const int sms=properties->multiProcessorCount;
    const int concurrency=temp_g.size(0);
    TORCH_CHECK(concurrency>0 && concurrency*MOE_SMS_PER_EXPERT<=sms, "Invalid expert concurrency");
    const int groups=MIN(MIN(concurrency,experts),MOE_MAX_GROUPS);
    const int width=MIN(sms/groups,MOE_MAX_SMS_PER_EXPERT);
    constexpr int threads=EXL3_GEMM_BASE_THREADS*MOE_TILESIZE_K/16;
    static std::once_flag configured[MAX_DEVICES];
    TORCH_CHECK(x.get_device()<MAX_DEVICES, "Device index exceeds native limit");
    std::call_once(configured[x.get_device()], [&] {
        C10_CUDA_CHECK(cudaFuncSetAttribute(sage_heterogeneous_kernel<0,256,2>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
        int active=0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,sage_heterogeneous_kernel<0,256,2>,threads,SMEM_MAX));
        TORCH_CHECK(active>=1, "Expert group barriers require a resident block on every selected SM");
    });
    auto* p=pointers.data_ptr<int64_t>();
    sage_heterogeneous_kernel<0,256,2><<<dim3(width,1,groups),threads,SMEM_MAX,at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<half*>(temp_g.data_ptr()),reinterpret_cast<half*>(temp_u.data_ptr()),
        reinterpret_cast<half*>(temp_ig.data_ptr()),reinterpret_cast<half*>(temp_iu.data_ptr()),out.data_ptr<float>(),
        reinterpret_cast<const uint16_t**>(p),reinterpret_cast<const half**>(p+experts),reinterpret_cast<const half**>(p+2*experts),
        reinterpret_cast<const uint16_t**>(p+3*experts),reinterpret_cast<const half**>(p+4*experts),reinterpret_cast<const half**>(p+5*experts),
        reinterpret_cast<const uint16_t**>(p+6*experts),reinterpret_cast<const half**>(p+7*experts),reinterpret_cast<const half**>(p+8*experts),
        counts.data_ptr<int64_t>(),tokens.data_ptr<int64_t>(),reinterpret_cast<const half*>(weights.data_ptr()),
        5120,2304,experts,tokens.size(0)/x.size(0),temp_g.size(1),groups,static_cast<float>(limit),MOE_ACT_SILU,
        bits.data_ptr<int64_t>(),locks.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
