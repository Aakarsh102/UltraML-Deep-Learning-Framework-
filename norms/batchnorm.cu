#include "norms.h"
#include "../core/macros.h"

#include <cmath>
#include <cstdio>

namespace ultraml {

// ============================================================================
// BatchNorm1d: input [B, F]. One block per feature column.
// ============================================================================
namespace {

__global__ void bn1d_stats(const float* x, float* mean, float* var,
                           int B, int F) {
    int j = blockIdx.x;
    if (j >= F) return;
    extern __shared__ float smem[];   // 2 * blockDim.x
    float* s_sum  = smem;
    float* s_sum2 = smem + blockDim.x;

    int tid = threadIdx.x;
    float sum = 0.0f, sum2 = 0.0f;
    for (int i = tid; i < B; i += blockDim.x) {
        float v = x[i * F + j];
        sum  += v;
        sum2 += v * v;
    }
    s_sum[tid]  = sum;
    s_sum2[tid] = sum2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { s_sum[tid] += s_sum[tid+s]; s_sum2[tid] += s_sum2[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) {
        float m = s_sum[0] / B;
        mean[j] = m;
        var[j]  = s_sum2[0] / B - m * m;
    }
}

__global__ void bn1d_apply(const float* x, const float* mean, const float* inv_std,
                           const float* gamma, const float* beta,
                           float* y, int B, int F) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * F;
    if (idx >= total) return;
    int j = idx % F;
    float xhat = (x[idx] - mean[j]) * inv_std[j];
    y[idx] = gamma[j] * xhat + beta[j];
}

__global__ void bn1d_inv_std(const float* var, float* inv_std, int F, float eps) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < F) inv_std[j] = rsqrtf(var[j] + eps);
}

__global__ void bn1d_update_running(const float* mean, const float* var,
                                    float* running_mean, float* running_var,
                                    int F, float momentum) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < F) {
        running_mean[j] = (1.0f - momentum) * running_mean[j] + momentum * mean[j];
        running_var [j] = (1.0f - momentum) * running_var [j] + momentum * var [j];
    }
}

// Per-feature sums needed for backward: sum(dy) and sum(dy * xhat)
__global__ void bn1d_reduce_dy(const float* dy, const float* x,
                               const float* mean, const float* inv_std,
                               float* sum_dy, float* sum_dy_xhat,
                               int B, int F) {
    int j = blockIdx.x;
    if (j >= F) return;
    extern __shared__ float smem[];
    float* s1 = smem;
    float* s2 = smem + blockDim.x;

    int tid = threadIdx.x;
    float a = 0.0f, b = 0.0f;
    for (int i = tid; i < B; i += blockDim.x) {
        float g  = dy[i * F + j];
        float xh = (x[i * F + j] - mean[j]) * inv_std[j];
        a += g;
        b += g * xh;
    }
    s1[tid] = a; s2[tid] = b;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { s1[tid] += s1[tid+s]; s2[tid] += s2[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) {
        sum_dy     [j] = s1[0];
        sum_dy_xhat[j] = s2[0];
        // grad_beta  = sum_dy
        // grad_gamma = sum_dy_xhat   -- caller copies these out
    }
}

__global__ void bn1d_dx(const float* dy, const float* x,
                        const float* mean, const float* inv_std,
                        const float* gamma,
                        const float* sum_dy, const float* sum_dy_xhat,
                        float* dx, int B, int F) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * F;
    if (idx >= total) return;
    int j = idx % F;

    float xhat = (x[idx] - mean[j]) * inv_std[j];
    float inv_B = 1.0f / (float)B;
    float term = dy[idx] - inv_B * sum_dy[j] - xhat * inv_B * sum_dy_xhat[j];
    dx[idx] = gamma[j] * inv_std[j] * term;
}

} // namespace

void batchnorm1d_forward(const Tensor* input,
                         const Tensor* gamma,
                         const Tensor* beta,
                         Tensor* running_mean,
                         Tensor* running_var,
                         Tensor* saved_mean,
                         Tensor* saved_inv_std,
                         Tensor* output,
                         float momentum, float eps, bool training) {
    int B = input->shape[0];
    int F = input->shape[1];

    float *mean_ptr, *var_ptr, *inv_std_ptr;
    float *tmp_mean = nullptr, *tmp_var = nullptr, *tmp_inv = nullptr;

    if (training) {
        if (saved_mean)    mean_ptr    = saved_mean->data;
        else               { ULTRAML_CUDA_CHECK(cudaMalloc(&tmp_mean, F*sizeof(float))); mean_ptr = tmp_mean; }
        ULTRAML_CUDA_CHECK(cudaMalloc(&tmp_var, F*sizeof(float)));
        var_ptr = tmp_var;
        if (saved_inv_std) inv_std_ptr = saved_inv_std->data;
        else               { ULTRAML_CUDA_CHECK(cudaMalloc(&tmp_inv, F*sizeof(float))); inv_std_ptr = tmp_inv; }

        int threads = 256;
        size_t smem = 2 * threads * sizeof(float);
        bn1d_stats<<<F, threads, smem>>>(input->data, mean_ptr, var_ptr, B, F);

        int t2 = 256;
        int b2 = (F + t2 - 1) / t2;
        bn1d_inv_std<<<b2, t2>>>(var_ptr, inv_std_ptr, F, eps);
        bn1d_update_running<<<b2, t2>>>(mean_ptr, var_ptr,
                                        running_mean->data, running_var->data,
                                        F, momentum);
    } else {
        // eval: use running stats and compute inv_std from running_var
        ULTRAML_CUDA_CHECK(cudaMalloc(&tmp_inv, F*sizeof(float)));
        inv_std_ptr = tmp_inv;
        mean_ptr    = running_mean->data;
        int t2 = 256, b2 = (F + t2 - 1) / t2;
        bn1d_inv_std<<<b2, t2>>>(running_var->data, inv_std_ptr, F, eps);
    }

    int t = 256;
    int total = B * F;
    int b = (total + t - 1) / t;
    bn1d_apply<<<b, t>>>(input->data, mean_ptr, inv_std_ptr,
                         gamma->data, beta->data, output->data, B, F);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());

    if (tmp_mean) ULTRAML_CUDA_CHECK(cudaFree(tmp_mean));
    if (tmp_var)  ULTRAML_CUDA_CHECK(cudaFree(tmp_var));
    if (tmp_inv && (!saved_inv_std || !training)) ULTRAML_CUDA_CHECK(cudaFree(tmp_inv));
}

void batchnorm1d_backward(const Tensor* grad_output,
                          const Tensor* input,
                          const Tensor* gamma,
                          const Tensor* saved_mean,
                          const Tensor* saved_inv_std,
                          Tensor* grad_input,
                          Tensor* grad_gamma,
                          Tensor* grad_beta) {
    int B = input->shape[0];
    int F = input->shape[1];

    float *sum_dy = nullptr, *sum_dy_xhat = nullptr;
    ULTRAML_CUDA_CHECK(cudaMalloc(&sum_dy,      F*sizeof(float)));
    ULTRAML_CUDA_CHECK(cudaMalloc(&sum_dy_xhat, F*sizeof(float)));

    int threads = 256;
    size_t smem = 2 * threads * sizeof(float);
    bn1d_reduce_dy<<<F, threads, smem>>>(grad_output->data, input->data,
                                         saved_mean->data, saved_inv_std->data,
                                         sum_dy, sum_dy_xhat, B, F);

    if (grad_beta)
        ULTRAML_CUDA_CHECK(cudaMemcpy(grad_beta->data, sum_dy,
                                      F*sizeof(float), cudaMemcpyDeviceToDevice));
    if (grad_gamma)
        ULTRAML_CUDA_CHECK(cudaMemcpy(grad_gamma->data, sum_dy_xhat,
                                      F*sizeof(float), cudaMemcpyDeviceToDevice));

    int t2 = 256;
    int total = B * F;
    int b2 = (total + t2 - 1) / t2;
    bn1d_dx<<<b2, t2>>>(grad_output->data, input->data,
                        saved_mean->data, saved_inv_std->data,
                        gamma->data, sum_dy, sum_dy_xhat,
                        grad_input->data, B, F);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());

    ULTRAML_CUDA_CHECK(cudaFree(sum_dy));
    ULTRAML_CUDA_CHECK(cudaFree(sum_dy_xhat));
}

// ============================================================================
// BatchNorm2d: input [B, C, H, W]. One block per channel, reduce over B*H*W.
// ============================================================================
namespace {

__global__ void bn2d_stats(const float* x, float* mean, float* var,
                           int B, int C, int HW) {
    int c = blockIdx.x;
    if (c >= C) return;
    extern __shared__ float smem[];
    float* s1 = smem;
    float* s2 = smem + blockDim.x;

    int tid = threadIdx.x;
    float sum = 0.0f, sum2 = 0.0f;
    int N = B * HW;
    for (int k = tid; k < N; k += blockDim.x) {
        int b  = k / HW;
        int p  = k % HW;
        float v = x[((b * C) + c) * HW + p];
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
        mean[c] = m;
        var [c] = s2[0] / (float)N - m * m;
    }
}

__global__ void bn2d_apply(const float* x, const float* mean, const float* inv_std,
                           const float* gamma, const float* beta,
                           float* y, int B, int C, int HW) {
    int total = B * C * HW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = (idx / HW) % C;
    float xhat = (x[idx] - mean[c]) * inv_std[c];
    y[idx] = gamma[c] * xhat + beta[c];
}

__global__ void bn2d_reduce_dy(const float* dy, const float* x,
                               const float* mean, const float* inv_std,
                               float* sum_dy, float* sum_dy_xhat,
                               int B, int C, int HW) {
    int c = blockIdx.x;
    if (c >= C) return;
    extern __shared__ float smem[];
    float* s1 = smem;
    float* s2 = smem + blockDim.x;
    int tid = threadIdx.x;
    int N = B * HW;
    float a = 0.0f, b = 0.0f;
    for (int k = tid; k < N; k += blockDim.x) {
        int bi = k / HW;
        int p  = k % HW;
        int off = ((bi * C) + c) * HW + p;
        float g  = dy[off];
        float xh = (x[off] - mean[c]) * inv_std[c];
        a += g; b += g * xh;
    }
    s1[tid] = a; s2[tid] = b;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { s1[tid] += s1[tid+s]; s2[tid] += s2[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) { sum_dy[c] = s1[0]; sum_dy_xhat[c] = s2[0]; }
}

__global__ void bn2d_dx(const float* dy, const float* x,
                        const float* mean, const float* inv_std,
                        const float* gamma,
                        const float* sum_dy, const float* sum_dy_xhat,
                        float* dx, int B, int C, int HW) {
    int total = B * C * HW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = (idx / HW) % C;
    int N = B * HW;
    float xhat = (x[idx] - mean[c]) * inv_std[c];
    float inv_N = 1.0f / (float)N;
    float term = dy[idx] - inv_N * sum_dy[c] - xhat * inv_N * sum_dy_xhat[c];
    dx[idx] = gamma[c] * inv_std[c] * term;
}

} // namespace

void batchnorm2d_forward(const Tensor* input,
                         const Tensor* gamma,
                         const Tensor* beta,
                         Tensor* running_mean,
                         Tensor* running_var,
                         Tensor* saved_mean,
                         Tensor* saved_inv_std,
                         Tensor* output,
                         float momentum, float eps, bool training) {
    int B  = input->shape[0];
    int C  = input->shape[1];
    int HW = input->shape[2] * input->shape[3];

    float *mean_ptr, *inv_std_ptr;
    float *tmp_mean = nullptr, *tmp_var = nullptr, *tmp_inv = nullptr;

    if (training) {
        if (saved_mean) {
            mean_ptr = saved_mean->data;
        } else {
            ULTRAML_CUDA_CHECK(cudaMalloc(&tmp_mean, C*sizeof(float)));
            mean_ptr = tmp_mean;
        }
        ULTRAML_CUDA_CHECK(cudaMalloc(&tmp_var, C*sizeof(float)));
        if (saved_inv_std) {
            inv_std_ptr = saved_inv_std->data;
        } else {
            ULTRAML_CUDA_CHECK(cudaMalloc(&tmp_inv, C*sizeof(float)));
            inv_std_ptr = tmp_inv;
        }

        int threads = 256;
        size_t smem = 2 * threads * sizeof(float);
        bn2d_stats<<<C, threads, smem>>>(input->data, mean_ptr, tmp_var, B, C, HW);
        int t2 = 256, b2 = (C + t2 - 1) / t2;
        bn1d_inv_std<<<b2, t2>>>(tmp_var, inv_std_ptr, C, eps);
        bn1d_update_running<<<b2, t2>>>(mean_ptr, tmp_var,
                                        running_mean->data, running_var->data,
                                        C, momentum);
        ULTRAML_CUDA_CHECK(cudaFree(tmp_var));
    } else {
        ULTRAML_CUDA_CHECK(cudaMalloc(&tmp_inv, C*sizeof(float)));
        inv_std_ptr = tmp_inv;
        mean_ptr    = running_mean->data;
        int t2 = 256, b2 = (C + t2 - 1) / t2;
        bn1d_inv_std<<<b2, t2>>>(running_var->data, inv_std_ptr, C, eps);
    }

    int total = B * C * HW;
    int t = 256;
    int b = (total + t - 1) / t;
    bn2d_apply<<<b, t>>>(input->data, mean_ptr, inv_std_ptr,
                         gamma->data, beta->data, output->data, B, C, HW);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());

    if (tmp_mean) ULTRAML_CUDA_CHECK(cudaFree(tmp_mean));
    if (tmp_inv && (!saved_inv_std || !training)) ULTRAML_CUDA_CHECK(cudaFree(tmp_inv));
}

void batchnorm2d_backward(const Tensor* grad_output,
                          const Tensor* input,
                          const Tensor* gamma,
                          const Tensor* saved_mean,
                          const Tensor* saved_inv_std,
                          Tensor* grad_input,
                          Tensor* grad_gamma,
                          Tensor* grad_beta) {
    int B  = input->shape[0];
    int C  = input->shape[1];
    int HW = input->shape[2] * input->shape[3];

    float *sum_dy, *sum_dy_xhat;
    ULTRAML_CUDA_CHECK(cudaMalloc(&sum_dy,      C*sizeof(float)));
    ULTRAML_CUDA_CHECK(cudaMalloc(&sum_dy_xhat, C*sizeof(float)));

    int threads = 256;
    size_t smem = 2 * threads * sizeof(float);
    bn2d_reduce_dy<<<C, threads, smem>>>(grad_output->data, input->data,
                                         saved_mean->data, saved_inv_std->data,
                                         sum_dy, sum_dy_xhat, B, C, HW);

    if (grad_beta)
        ULTRAML_CUDA_CHECK(cudaMemcpy(grad_beta->data, sum_dy,
                                      C*sizeof(float), cudaMemcpyDeviceToDevice));
    if (grad_gamma)
        ULTRAML_CUDA_CHECK(cudaMemcpy(grad_gamma->data, sum_dy_xhat,
                                      C*sizeof(float), cudaMemcpyDeviceToDevice));

    int total = B * C * HW;
    int t2 = 256, b2 = (total + t2 - 1) / t2;
    bn2d_dx<<<b2, t2>>>(grad_output->data, input->data,
                        saved_mean->data, saved_inv_std->data,
                        gamma->data, sum_dy, sum_dy_xhat,
                        grad_input->data, B, C, HW);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());

    ULTRAML_CUDA_CHECK(cudaFree(sum_dy));
    ULTRAML_CUDA_CHECK(cudaFree(sum_dy_xhat));
}

} // namespace ultraml
