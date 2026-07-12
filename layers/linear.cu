#include "layers.h"
#include "../core/macros.h"

namespace ultraml {

namespace {

// Adds a per-feature bias to a row-major [batch, features] output.
__global__ void add_bias_kernel(float* out, const float* bias,
                                int batch, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * features;
    if (idx < total) out[idx] += bias[idx % features];
}

// Reduces grad_output [batch, features] along batch dim into grad_bias[features].
// One block per feature; threads cooperatively sum the column.
__global__ void sum_bias_kernel(const float* grad_out, float* grad_bias,
                                int batch, int features) {
    int f = blockIdx.x;
    if (f >= features) return;

    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float s = 0.0f;
    for (int i = tid; i < batch; i += blockDim.x) s += grad_out[i * features + f];
    smem[tid] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) grad_bias[f] = smem[0];
}

} // namespace

void linear_forward(NNContext* ctx,
                    const Tensor* input,
                    const Tensor* weight,
                    const Tensor* bias,
                    Tensor* output) {
    int B = input->shape[0];
    int I = input->shape[1];
    int O = weight->shape[0];

    const float alpha = 1.0f, beta = 0.0f;
    // row-major Y = X @ W^T  ==  column-major out = W^T * X
    // M=O, N=B, K=I ; A=W (lda=I, op=T), B=X (ldb=I, op=N), C=Y (ldc=O)
    ULTRAML_CUBLAS_CHECK(cublasSgemm(ctx->cublas_handle,
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     O, B, I,
                                     &alpha,
                                     weight->data, I,
                                     input->data,  I,
                                     &beta,
                                     output->data, O));

    if (bias) {
        int total = B * O;
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        add_bias_kernel<<<blocks, threads>>>(output->data, bias->data, B, O);
        ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void linear_backward(NNContext* ctx,
                     const Tensor* grad_output,
                     const Tensor* input,
                     const Tensor* weight,
                     Tensor* grad_input,
                     Tensor* grad_weight,
                     Tensor* grad_bias) {
    int B = input->shape[0];
    int I = input->shape[1];
    int O = weight->shape[0];

    const float alpha = 1.0f, beta = 0.0f;

    // grad_input = grad_output @ weight   (row-major)
    // column-major: G_in = W * G_out ;  M=I, N=B, K=O ; W (lda=I, N), G_out (ldb=O, N)
    if (grad_input) {
        ULTRAML_CUBLAS_CHECK(cublasSgemm(ctx->cublas_handle,
                                         CUBLAS_OP_N, CUBLAS_OP_N,
                                         I, B, O,
                                         &alpha,
                                         weight->data,      I,
                                         grad_output->data, O,
                                         &beta,
                                         grad_input->data,  I));
    }

    // grad_weight = grad_output^T @ input   (row-major)
    // column-major: G_w = X * G_out^T ;  M=I, N=O, K=B ; X (lda=I, N), G_out (ldb=O, T)
    if (grad_weight) {
        ULTRAML_CUBLAS_CHECK(cublasSgemm(ctx->cublas_handle,
                                         CUBLAS_OP_N, CUBLAS_OP_T,
                                         I, O, B,
                                         &alpha,
                                         input->data,       I,
                                         grad_output->data, O,
                                         &beta,
                                         grad_weight->data, I));
    }

    // grad_bias = sum(grad_output, dim=0)
    if (grad_bias) {
        int threads = 256;
        size_t smem = threads * sizeof(float);
        sum_bias_kernel<<<O, threads, smem>>>(grad_output->data, grad_bias->data,
                                              B, O);
        ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// ============ element-wise utilities =================================
namespace {
__global__ void k_add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
__global__ void k_mul(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}
__global__ void k_scale(const float* in, float s, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * s;
}
__global__ void k_axpy(const float* x, float a, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}
} // namespace

void tensor_add(const Tensor* a, const Tensor* b, Tensor* out) {
    int t = 256, blk = (a->size + t - 1) / t;
    k_add<<<blk, t>>>(a->data, b->data, out->data, a->size);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}
void tensor_multiply(const Tensor* a, const Tensor* b, Tensor* out) {
    int t = 256, blk = (a->size + t - 1) / t;
    k_mul<<<blk, t>>>(a->data, b->data, out->data, a->size);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}
void tensor_scale(const Tensor* in, float s, Tensor* out) {
    int t = 256, blk = (in->size + t - 1) / t;
    k_scale<<<blk, t>>>(in->data, s, out->data, in->size);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}
void tensor_axpy(const Tensor* x, float a, Tensor* y) {
    int t = 256, blk = (x->size + t - 1) / t;
    k_axpy<<<blk, t>>>(x->data, a, y->data, x->size);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace ultraml
