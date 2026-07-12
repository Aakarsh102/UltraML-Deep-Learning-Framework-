#include "norms.h"
#include "../core/macros.h"

#include <cmath>
#include <cstdio>

namespace ultraml {

// GroupNorm: input [B, C, H, W]. Channels split into G groups (C must be
// divisible by G). Statistics computed per (sample, group) over
// (C/G * H * W) elements. gamma/beta are per-channel.

namespace {

__global__ void gn_stats(const float* x, float* mean, float* inv_std,
                         int B, int C, int HW, int G, float eps) {
    int b = blockIdx.x / G;
    int g = blockIdx.x % G;
    int cpg = C / G;
    int N = cpg * HW;

    extern __shared__ float smem[];
    float* s1 = smem;
    float* s2 = smem + blockDim.x;

    int tid = threadIdx.x;
    float sum = 0.0f, sum2 = 0.0f;
    for (int k = tid; k < N; k += blockDim.x) {
        int c_in_g = k / HW;
        int p      = k % HW;
        int c      = g * cpg + c_in_g;
        float v    = x[((b * C) + c) * HW + p];
        sum  += v;
        sum2 += v * v;
    }
    s1[tid] = sum; s2[tid] = sum2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { s1[tid] += s1[tid+s]; s2[tid] += s2[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) {
        float m = s1[0] / (float)N;
        float v = s2[0] / (float)N - m * m;
        mean[b*G + g]    = m;
        inv_std[b*G + g] = rsqrtf(v + eps);
    }
}

__global__ void gn_apply(const float* x, const float* mean, const float* inv_std,
                         const float* gamma, const float* beta, float* y,
                         int B, int C, int HW, int G) {
    int total = B * C * HW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int p = idx % HW;
    int c = (idx / HW) % C;
    int b = idx / (C * HW);
    int g = c / (C / G);
    float xhat = (x[idx] - mean[b*G + g]) * inv_std[b*G + g];
    y[idx] = gamma[c] * xhat + beta[c];
}

// Per-(b,g) sums of dxhat and dxhat*xhat for dx computation.
__global__ void gn_reduce_dxhat(const float* dy, const float* x,
                                const float* gamma,
                                const float* mean, const float* inv_std,
                                float* sum_dxhat, float* sum_dxhat_xhat,
                                int B, int C, int HW, int G) {
    int b = blockIdx.x / G;
    int g = blockIdx.x % G;
    int cpg = C / G;
    int N = cpg * HW;

    extern __shared__ float smem[];
    float* s1 = smem;
    float* s2 = smem + blockDim.x;
    int tid = threadIdx.x;
    float m  = mean[b*G + g];
    float is = inv_std[b*G + g];

    float a = 0.0f, bb = 0.0f;
    for (int k = tid; k < N; k += blockDim.x) {
        int c_in_g = k / HW;
        int p      = k % HW;
        int c      = g * cpg + c_in_g;
        int off    = ((b * C) + c) * HW + p;
        float xhat  = (x[off] - m) * is;
        float dxhat = dy[off] * gamma[c];
        a  += dxhat;
        bb += dxhat * xhat;
    }
    s1[tid] = a; s2[tid] = bb;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { s1[tid] += s1[tid+s]; s2[tid] += s2[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) {
        sum_dxhat     [b*G + g] = s1[0];
        sum_dxhat_xhat[b*G + g] = s2[0];
    }
}

__global__ void gn_dx(const float* dy, const float* x,
                      const float* gamma,
                      const float* mean, const float* inv_std,
                      const float* sum_dxhat, const float* sum_dxhat_xhat,
                      float* dx, int B, int C, int HW, int G) {
    int total = B * C * HW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int p = idx % HW;
    int c = (idx / HW) % C;
    int b = idx / (C * HW);
    int g = c / (C / G);

    int cpg = C / G;
    int N   = cpg * HW;
    float m  = mean[b*G + g];
    float is = inv_std[b*G + g];

    float xhat  = (x[idx] - m) * is;
    float dxhat = dy[idx] * gamma[c];
    float inv_N = 1.0f / (float)N;
    float term  = dxhat - inv_N * sum_dxhat[b*G + g]
                        - xhat * inv_N * sum_dxhat_xhat[b*G + g];
    dx[idx] = is * term;
}

// grad_gamma[c] = sum over (b, HW) of dy * xhat
// grad_beta [c] = sum over (b, HW) of dy
__global__ void gn_bwd_params(const float* dy, const float* x,
                              const float* mean, const float* inv_std,
                              float* grad_gamma, float* grad_beta,
                              int B, int C, int HW, int G) {
    int c = blockIdx.x;
    if (c >= C) return;
    int g = c / (C / G);

    extern __shared__ float smem[];
    float* s1 = smem;
    float* s2 = smem + blockDim.x;
    int tid = threadIdx.x;

    float gg = 0.0f, gb = 0.0f;
    int N = B * HW;
    for (int k = tid; k < N; k += blockDim.x) {
        int bi = k / HW;
        int p  = k % HW;
        int off = ((bi * C) + c) * HW + p;
        float m  = mean[bi*G + g];
        float is = inv_std[bi*G + g];
        float xhat = (x[off] - m) * is;
        gg += dy[off] * xhat;
        gb += dy[off];
    }
    s1[tid] = gg; s2[tid] = gb;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { s1[tid] += s1[tid+s]; s2[tid] += s2[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) {
        if (grad_gamma) grad_gamma[c] = s1[0];
        if (grad_beta ) grad_beta [c] = s2[0];
    }
}

} // namespace

void groupnorm_forward(const Tensor* input, const Tensor* gamma, const Tensor* beta,
                       int num_groups,
                       Tensor* saved_mean, Tensor* saved_inv_std,
                       Tensor* output, float eps) {
    int B = input->shape[0];
    int C = input->shape[1];
    int HW = input->shape[2] * input->shape[3];
    if (C % num_groups != 0) {
        fprintf(stderr, "groupnorm: C (%d) must be divisible by num_groups (%d)\n",
                C, num_groups);
        exit(EXIT_FAILURE);
    }
    int G = num_groups;

    int threads = 256;
    size_t smem = 2 * threads * sizeof(float);
    gn_stats<<<B * G, threads, smem>>>(input->data,
                                       saved_mean->data, saved_inv_std->data,
                                       B, C, HW, G, eps);

    int total = B * C * HW;
    int t = 256, b = (total + t - 1) / t;
    gn_apply<<<b, t>>>(input->data, saved_mean->data, saved_inv_std->data,
                       gamma->data, beta->data, output->data, B, C, HW, G);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

void groupnorm_backward(const Tensor* grad_output, const Tensor* input,
                        const Tensor* gamma, int num_groups,
                        const Tensor* saved_mean, const Tensor* saved_inv_std,
                        Tensor* grad_input, Tensor* grad_gamma, Tensor* grad_beta) {
    int B = input->shape[0];
    int C = input->shape[1];
    int HW = input->shape[2] * input->shape[3];
    int G = num_groups;

    float *sum_dxhat = nullptr, *sum_dxhat_xhat = nullptr;
    ULTRAML_CUDA_CHECK(cudaMalloc(&sum_dxhat,      B*G*sizeof(float)));
    ULTRAML_CUDA_CHECK(cudaMalloc(&sum_dxhat_xhat, B*G*sizeof(float)));

    int threads = 256;
    size_t smem = 2 * threads * sizeof(float);
    gn_reduce_dxhat<<<B * G, threads, smem>>>(
        grad_output->data, input->data, gamma->data,
        saved_mean->data, saved_inv_std->data,
        sum_dxhat, sum_dxhat_xhat, B, C, HW, G);

    int total = B * C * HW;
    int t = 256, b = (total + t - 1) / t;
    gn_dx<<<b, t>>>(grad_output->data, input->data, gamma->data,
                    saved_mean->data, saved_inv_std->data,
                    sum_dxhat, sum_dxhat_xhat,
                    grad_input->data, B, C, HW, G);

    int t_c = 256;
    size_t smem_c = 2 * t_c * sizeof(float);
    gn_bwd_params<<<C, t_c, smem_c>>>(
        grad_output->data, input->data,
        saved_mean->data, saved_inv_std->data,
        grad_gamma ? grad_gamma->data : nullptr,
        grad_beta  ? grad_beta ->data : nullptr,
        B, C, HW, G);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());

    ULTRAML_CUDA_CHECK(cudaFree(sum_dxhat));
    ULTRAML_CUDA_CHECK(cudaFree(sum_dxhat_xhat));
}

} // namespace ultraml
