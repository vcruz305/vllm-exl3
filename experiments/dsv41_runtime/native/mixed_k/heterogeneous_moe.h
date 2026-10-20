#pragma once
#include <ATen/Tensor.h>
void sage_heterogeneous_moe(
    const at::Tensor& x, const at::Tensor& out, const at::Tensor& counts,
    const at::Tensor& tokens, const at::Tensor& weights,
    const at::Tensor& temp_g, const at::Tensor& temp_u,
    const at::Tensor& temp_ig, const at::Tensor& temp_iu,
    const at::Tensor& bits, const at::Tensor& pointers,
    const at::Tensor& locks, double limit);
