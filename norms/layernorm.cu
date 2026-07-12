#include "norms.h"
#include "../core/macros.h"

#include <cmath>

namespace ultraml {

// LayerNorm normalizes over the last axis. The caller flattens any leading
// dims into "rows", leaving cols = input->shape[ndim-1].

namespace {

__global__ void ln_fwd(const float* x, const float* gamma, const float* beta,
                       float* mean, float* inv_std, float* y,
                       int rows, int cols, float eps) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* xr = x + r * cols;
    float*       yr = y + r * cols;

    extern __shared__ float smem[];
    float* s1 = smem;
    float* s2 = smem + blockDim.x;
    int tid = threadIdx.x;

    float sum = 0.0f, sum2 = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float v = xr[j];
        sum  += v;
        sum2 += v * v;
    }
    s1[tid] = sum; s2[tid] = sum2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { s1[tid] += s1[tid+s]; s2[tid] += s2[tid+s]; }
        __syncthreads();
    }
    float m = s1[0] / (float)cols;
    float v = s2[0] / (float)cols - m * m;
    float is = rsqrtf(v + eps);
    if (tid == 0) { mean[r] = m; inv_std[r] = is; }

    for (int j = tid; j < cols; j += blockDim.x) {
        float xhat = (xr[j] - m) * is;
        yr[j] = gamma[j] * xhat + beta[j];
    }
}

// Per-row backward: dx[j] = inv_std * (dxhat[j] - mean(dxhat) - xhat[j]*mean(dxhat*xhat))
__global__ void ln_bwd_dx(const float* dy, const float* x,
                          const float* gamma,
                          const float* mean, const float* inv_std,
                          float* dx, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* dyr = dy + r * cols;
    const float* xr  = x  + r * cols;
    float*       dxr = dx + r * cols;

    extern __shared__ float smem[];
    float* s1 = smem;
    float* s2 = smem + blockDim.x;
    int tid = threadIdx.x;

    float m  = mean[r];
    float is = inv_std[r];

    float a = 0.0f, b = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float xhat  = (xr[j] - m) * is;
        float dxhat = dyr[j] * gamma[j];
        a += dxhat;
        b += dxhat * xhat;
    }
    s1[tid] = a; s2[tid] = b;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { s1[tid] += s1[tid+s]; s2[tid] += s2[tid+s]; }
        __syncthreads();
    }
    float mean_dxhat      = s1[0] / (float)cols;
    float mean_dxhat_xhat = s2[0] / (float)cols;

    for (int j = tid; j < cols; j += blockDim.x) {
        float xhat  = (xr[j] - m) * is;
        float dxhat = dyr[j] * gamma[j];
        dxr[j] = is * (dxhat - mean_dxhat - xhat * mean_dxhat_xhat);
    }
}

// grad_gamma[j] = sum_i dy[i,j] * xhat[i,j]
// grad_beta [j] = sum_i dy[i,j]
__global__ void ln_bwd_params(const float* dy, const float* x,
                              const float* mean, const float* inv_std,
                              float* grad_gamma, float* grad_beta,
                              int rows, int cols) {
    int j = blockIdx.x;
    if (j >= cols) return;
    extern __shared__ float smem[];
    float* s1 = smem;
    float* s2 = smem + blockDim.x;
    int tid = threadIdx.x;

    float gg = 0.0f, gb = 0.0f;
    for (int i = tid; i < rows; i += blockDim.x) {
        float g    = dy[i * cols + j];
        float xhat = (x[i * cols + j] - mean[i]) * inv_std[i];
        gg += g * xhat;
        gb += g;
    }
    s1[tid] = gg; s2[tid] = gb;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { s1[tid] += s1[tid+s]; s2[tid] += s2[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) {
        if (grad_gamma) grad_gamma[j] = s1[0];
        if (grad_beta ) grad_beta [j] = s2[0];
    }
}

inline int pick_threads(int n) {
    int t = 1;
    while (t < n && t < 512) t <<= 1;
    return t < 32 ? 32 : t;
}

inline void row_col_flatten(const Tensor* t, int& rows, int& cols) {
    cols = t->shape[t->ndim - 1];
    rows = t->size / cols;
}

} // namespace

void layernorm_forward(const Tensor* input, const Tensor* gamma, const Tensor* beta,
                       Tensor* saved_mean, Tensor* saved_inv_std,
                       Tensor* output, float eps) {
    int rows, cols;
    row_col_flatten(input, rows, cols);
    int threads = pick_threads(cols);
    size_t smem = 2 * threads * sizeof(float);
    ln_fwd<<<rows, threads, smem>>>(input->data, gamma->data, beta->data,
                                    saved_mean->data, saved_inv_std->data,
                                    output->data, rows, cols, eps);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

void layernorm_backward(const Tensor* grad_output, const Tensor* input,
                        const Tensor* gamma,
                        const Tensor* saved_mean, const Tensor* saved_inv_std,
                        Tensor* grad_input, Tensor* grad_gamma, Tensor* grad_beta) {
    int rows, cols;
    row_col_flatten(input, rows, cols);

    int t_r = pick_threads(cols);
    size_t smem_r = 2 * t_r * sizeof(float);
    ln_bwd_dx<<<rows, t_r, smem_r>>>(grad_output->data, input->data, gamma->data,
                                     saved_mean->data, saved_inv_std->data,
                                     grad_input->data, rows, cols);

    int t_c = pick_threads(rows);
    size_t smem_c = 2 * t_c * sizeof(float);
    ln_bwd_params<<<cols, t_c, smem_c>>>(
        grad_output->data, input->data,
        saved_mean->data, saved_inv_std->data,
        grad_gamma ? grad_gamma->data : nullptr,
        grad_beta  ? grad_beta ->data : nullptr,
        rows, cols);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace ultraml
