#pragma once

#include "../core/tensor.h"

// Normalization layers. Each forward takes (input, params...) and writes
// output. Training-mode forwards that need intermediate statistics for the
// backward pass expose `saved_mean`/`saved_inv_std` / `saved_rrms` tensors
// which the caller allocates once and passes to backward(). This "explicit
// saved state" pattern is what a future autograd engine will record.
namespace ultraml {

// ==================== BatchNorm1d ====================================
// Input [batch, features]. gamma/beta/running_mean/running_var are
// [features]. In training mode, per-batch mean & var are computed and the
// running stats updated. saved_mean/saved_inv_std (both [features]) are
// written for the backward pass.
void batchnorm1d_forward(const Tensor* input,
                         const Tensor* gamma,
                         const Tensor* beta,
                         Tensor* running_mean,
                         Tensor* running_var,
                         Tensor* saved_mean,      // may be nullptr in eval
                         Tensor* saved_inv_std,   // may be nullptr in eval
                         Tensor* output,
                         float  momentum,
                         float  eps,
                         bool   training);

void batchnorm1d_backward(const Tensor* grad_output,
                          const Tensor* input,
                          const Tensor* gamma,
                          const Tensor* saved_mean,
                          const Tensor* saved_inv_std,
                          Tensor* grad_input,
                          Tensor* grad_gamma,
                          Tensor* grad_beta);

// ==================== BatchNorm2d ====================================
// Input [batch, channels, H, W]. Statistics are computed per channel over
// (batch, H, W). Same contract as 1d.
void batchnorm2d_forward(const Tensor* input,
                         const Tensor* gamma,
                         const Tensor* beta,
                         Tensor* running_mean,
                         Tensor* running_var,
                         Tensor* saved_mean,
                         Tensor* saved_inv_std,
                         Tensor* output,
                         float  momentum,
                         float  eps,
                         bool   training);

void batchnorm2d_backward(const Tensor* grad_output,
                          const Tensor* input,
                          const Tensor* gamma,
                          const Tensor* saved_mean,
                          const Tensor* saved_inv_std,
                          Tensor* grad_input,
                          Tensor* grad_gamma,
                          Tensor* grad_beta);

// ==================== LayerNorm ======================================
// Normalizes over the *last* dim. Input can be [rows, cols] after flatten.
// gamma, beta: [cols]. Saves per-row mean and inv_std (both [rows]).
void layernorm_forward(const Tensor* input,
                       const Tensor* gamma,
                       const Tensor* beta,
                       Tensor* saved_mean,
                       Tensor* saved_inv_std,
                       Tensor* output,
                       float   eps);

void layernorm_backward(const Tensor* grad_output,
                        const Tensor* input,
                        const Tensor* gamma,
                        const Tensor* saved_mean,
                        const Tensor* saved_inv_std,
                        Tensor* grad_input,
                        Tensor* grad_gamma,
                        Tensor* grad_beta);

// ==================== RMSNorm ========================================
// y = gamma * x / sqrt(mean(x^2) + eps). No beta. Normalizes last dim.
// gamma: [cols]. saved_rrms (= 1 / sqrt(...)): [rows].
void rmsnorm_forward(const Tensor* input,
                     const Tensor* gamma,
                     Tensor* saved_rrms,
                     Tensor* output,
                     float   eps);

void rmsnorm_backward(const Tensor* grad_output,
                      const Tensor* input,
                      const Tensor* gamma,
                      const Tensor* saved_rrms,
                      Tensor* grad_input,
                      Tensor* grad_gamma);

// ==================== GroupNorm ======================================
// Input [batch, channels, H, W]. Channels are split into `num_groups`
// groups; statistics are computed per (sample, group). gamma/beta: [channels].
// Saves mean and inv_std, shaped [batch, num_groups].
void groupnorm_forward(const Tensor* input,
                       const Tensor* gamma,
                       const Tensor* beta,
                       int num_groups,
                       Tensor* saved_mean,
                       Tensor* saved_inv_std,
                       Tensor* output,
                       float   eps);

void groupnorm_backward(const Tensor* grad_output,
                        const Tensor* input,
                        const Tensor* gamma,
                        int num_groups,
                        const Tensor* saved_mean,
                        const Tensor* saved_inv_std,
                        Tensor* grad_input,
                        Tensor* grad_gamma,
                        Tensor* grad_beta);

} // namespace ultraml
