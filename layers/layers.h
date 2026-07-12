#pragma once

#include "../core/tensor.h"
#include "../core/context.h"

#include <cudnn.h>

namespace ultraml {

// ==================== LINEAR =========================================
// Y = X @ W^T + b
//   input  [batch, in_features]
//   weight [out_features, in_features]
//   bias   [out_features]   (may be nullptr)
//   output [batch, out_features]
void linear_forward(NNContext* ctx,
                    const Tensor* input,
                    const Tensor* weight,
                    const Tensor* bias,
                    Tensor* output);

void linear_backward(NNContext* ctx,
                     const Tensor* grad_output,
                     const Tensor* input,
                     const Tensor* weight,
                     Tensor* grad_input,     // may be nullptr
                     Tensor* grad_weight,    // may be nullptr
                     Tensor* grad_bias);     // may be nullptr

// ==================== CONV2D =========================================
struct ConvDescriptor {
    cudnnTensorDescriptor_t       input_desc;
    cudnnTensorDescriptor_t       output_desc;
    cudnnFilterDescriptor_t       filter_desc;
    cudnnConvolutionDescriptor_t  conv_desc;
    cudnnConvolutionFwdAlgo_t     fwd_algo;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
    void*  workspace;
    size_t workspace_size;
};

ConvDescriptor* create_conv_descriptor(NNContext* ctx,
                                       int batch, int in_channels,
                                       int in_h, int in_w,
                                       int out_channels,
                                       int kernel_h, int kernel_w,
                                       int stride_h, int stride_w,
                                       int pad_h, int pad_w);
void free_conv_descriptor(ConvDescriptor* desc);

void conv2d_forward(NNContext* ctx, ConvDescriptor* desc,
                    const Tensor* input, const Tensor* weight, const Tensor* bias,
                    Tensor* output);

void conv2d_backward(NNContext* ctx, ConvDescriptor* desc,
                     const Tensor* grad_output,
                     const Tensor* input, const Tensor* weight,
                     Tensor* grad_input, Tensor* grad_weight, Tensor* grad_bias);

// ==================== POOLING ========================================
struct PoolDescriptor {
    cudnnPoolingDescriptor_t pooling_desc;
    cudnnTensorDescriptor_t  input_desc;
    cudnnTensorDescriptor_t  output_desc;
};

PoolDescriptor* create_pool_descriptor(int batch, int channels,
                                       int in_h, int in_w,
                                       int window_h, int window_w,
                                       int stride_h, int stride_w,
                                       int pad_h, int pad_w,
                                       cudnnPoolingMode_t mode);
void free_pool_descriptor(PoolDescriptor* desc);

void pool2d_forward(NNContext* ctx, PoolDescriptor* desc,
                    const Tensor* input, Tensor* output);

void pool2d_backward(NNContext* ctx, PoolDescriptor* desc,
                     const Tensor* grad_output, const Tensor* output,
                     const Tensor* input, Tensor* grad_input);

// ==================== ELEMENT-WISE UTILITIES =========================
// These are shape-preserving; each input/output has the same size.
void tensor_add     (const Tensor* a, const Tensor* b, Tensor* out);     // out = a + b
void tensor_multiply(const Tensor* a, const Tensor* b, Tensor* out);     // out = a * b
void tensor_scale   (const Tensor* input, float scale, Tensor* out);     // out = scale * input
void tensor_axpy    (const Tensor* x, float alpha, Tensor* y);           // y += alpha * x

// ==================== BATCHED MATMUL =================================
// C = op(A) @ op(B), batched. A, B, C are 3D [batch, rows, cols] or 2D
// (treated as batch = 1). op(X) = X^T when the matching trans flag is set;
// the trans flags describe how the *physical* row-major tensor is used:
//   op(A): [M, K],  op(B): [K, N],  C: [batch, M, N]
void batched_matmul(NNContext* ctx,
                    const Tensor* A, const Tensor* B, Tensor* C,
                    bool trans_a, bool trans_b);

// Gradients of C = op(A) @ op(B) wrt the physical A and B. Either grad
// output may be nullptr. Implemented as two more batched_matmul calls.
void batched_matmul_backward(NNContext* ctx,
                             const Tensor* grad_output,
                             const Tensor* A, const Tensor* B,
                             bool trans_a, bool trans_b,
                             Tensor* grad_a, Tensor* grad_b);

// ==================== PERMUTE / ATTENTION MASK =======================
// [d0, d1, d2, d3] -> [d0, d2, d1, d3] (swap dims 1 and 2). The backward of
// this permute is the same permute applied to grad_output.
void permute_0213(const Tensor* input, Tensor* output);

// Adds -1e9 to every position above the diagonal of each trailing [T, T]
// tile (pre-softmax attention scores). Shape-preserving; the last dim of
// input must be seq_len and rows cycle through query positions mod seq_len.
// Since the mask is an additive constant, its backward is the identity.
void add_causal_mask(const Tensor* input, Tensor* output, int seq_len);

// ==================== DROPOUT ========================================
// Fills mask with 0 (dropped, probability p) or 1/(1-p) (kept) using a
// stateless counter-based RNG, so forward is out = in * mask and backward
// is grad_in = grad_out * mask (both via tensor_multiply).
void dropout_make_mask(Tensor* mask, float p, unsigned long long seed);

// ==================== EMBEDDING ======================================
// weight [vocab, dim]; indices_device: n int32 ids in [0, vocab) on the
// device. output [n, dim]. Backward scatter-adds rows of grad_output into
// grad_weight, which must be zero-filled by the caller beforehand.
void embedding_forward(const Tensor* weight, const int* indices_device,
                       int n_indices, Tensor* output);
void embedding_backward(const Tensor* grad_output, const int* indices_device,
                        int n_indices, Tensor* grad_weight);

} // namespace ultraml
