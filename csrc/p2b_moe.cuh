#pragma once
#include <torch/extension.h>

at::Tensor p2b_fused_moe_cuda(const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rw, int64_t kg, int64_t ku,
    int64_t kd, bool mcg, int64_t intermediate_size, float swiglu_limit);


// Mixed-K (ABI 4) host entry: per-expert int8 K tables + n_local sentinel.
at::Tensor p2b_fused_moe_mk_cuda(const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rw,
    const at::Tensor& kg_tab, const at::Tensor& ku_tab, const at::Tensor& kd_tab,
    int64_t n_local, bool mcg, int64_t intermediate_size, float swiglu_limit);
