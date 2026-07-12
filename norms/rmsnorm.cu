#include "norms.h"
#include "../core/macros.h"

#include <cmath>

namespace ultraml {

// RMSNorm: y = gamma * x / sqrt(mean(x^2) + eps). No mean subtraction, no beta.
// Operates on the last axis.

namespace {

__global__ void rms_fwd(const float* x, const float* gamma,
                        float* rrms, float* y,
                        int rows, int cols, float eps) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* xr = x + r * cols;
    float*       yr = y + r * cols;

    extern __shared__ float smem[];
    int tid = threadIdx.x;

    float s = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float v = xr[j];
        s += v * v;
    }
    smem[tid] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) smem[tid] += smem[tid+k];
        __syncthreads();
    }
    float inv = rsqrtf(smem[0] / (float)cols + eps);
    if (tid == 0) rrms[r] = inv;

    for (int j = tid; j < cols; j += blockDim.x) {
        yr[j] = gamma[j] * xr[j] * inv;
    }
}

// dx[j] = rrms * gamma[j] * dy[j]
//       - (x[j] / (cols * rrms_sq_inverse_something)) * sum_k(dy[k]*gamma[k]*x[k])
// derivation:
//   r = 1/sqrt(mean(x^2)+eps); y_j = gamma_j * x_j * r
//   d(r)/d(x_j) = -x_j * r^3 / cols
//   dy_j/dx_k = gamma_j * (delta_jk * r + x_j * dr/dx_k)
//             = gamma_j * delta_jk * r - gamma_j * x_j * x_k * r^3 / cols
//   dx_k = sum_j dL/dy_j * dy_j/dx_k
//        = r * gamma_k * dL/dy_k
//          - (x_k * r^3 / cols) * sum_j(dL/dy_j * gamma_j * x_j)
__global__ void rms_bwd_dx(const float* dy, const float* x,
                           const float* gamma, const float* rrms,
                           float* dx, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* dyr = dy + r * cols;
    const float* xr  = x  + r * cols;
    float*       dxr = dx + r * cols;

    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float inv = rrms[r];

    float s = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        s += dyr[j] * gamma[j] * xr[j];
    }
    smem[tid] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) smem[tid] += smem[tid+k];
        __syncthreads();
    }
    float dot = smem[0];
    float coef = dot * inv * inv * inv / (float)cols;

    for (int j = tid; j < cols; j += blockDim.x) {
        dxr[j] = inv * gamma[j] * dyr[j] - xr[j] * coef;
    }
}

__global__ void rms_bwd_gamma(const float* dy, const float* x,
                              const float* rrms, float* grad_gamma,
                              int rows, int cols) {
    int j = blockIdx.x;
    if (j >= cols) return;
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float s = 0.0f;
    for (int i = tid; i < rows; i += blockDim.x) {
        s += dy[i*cols + j] * x[i*cols + j] * rrms[i];
    }
    smem[tid] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) smem[tid] += smem[tid+k];
        __syncthreads();
    }
    if (tid == 0) grad_gamma[j] = smem[0];
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

void rmsnorm_forward(const Tensor* input, const Tensor* gamma,
                     Tensor* saved_rrms, Tensor* output, float eps) {
    int rows, cols;
    row_col_flatten(input, rows, cols);
    int threads = pick_threads(cols);
    size_t smem = threads * sizeof(float);
    rms_fwd<<<rows, threads, smem>>>(input->data, gamma->data,
                                     saved_rrms->data, output->data,
                                     rows, cols, eps);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

void rmsnorm_backward(const Tensor* grad_output, const Tensor* input,
                      const Tensor* gamma, const Tensor* saved_rrms,
                      Tensor* grad_input, Tensor* grad_gamma) {
    int rows, cols;
    row_col_flatten(input, rows, cols);

    int t_r = pick_threads(cols);
    size_t smem_r = t_r * sizeof(float);
    rms_bwd_dx<<<rows, t_r, smem_r>>>(grad_output->data, input->data,
                                      gamma->data, saved_rrms->data,
                                      grad_input->data, rows, cols);

    if (grad_gamma) {
        int t_c = pick_threads(rows);
        size_t smem_c = t_c * sizeof(float);
        rms_bwd_gamma<<<cols, t_c, smem_c>>>(grad_output->data, input->data,
                                             saved_rrms->data,
                                             grad_gamma->data, rows, cols);
    }
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace ultraml
