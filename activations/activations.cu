#include "activations.h"
#include "../core/macros.h"

#include <cmath>

namespace ultraml {

namespace {

constexpr float GELU_C1 = 0.7978845608028654f;   // sqrt(2/pi)
constexpr float GELU_C2 = 0.044715f;

__device__ __forceinline__ float sigmoidf_dev(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float softplusf_dev(float x) {
    // numerically stable log1p(exp(x))
    return x > 0.0f ? x + log1pf(expf(-x)) : log1pf(expf(x));
}

// ============ kernels ================================================

__global__ void k_relu_fwd(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(0.0f, x[i]);
}
__global__ void k_relu_bwd(const float* dy, const float* x, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = (x[i] > 0.0f) ? dy[i] : 0.0f;
}

__global__ void k_lrelu_fwd(const float* x, float* y, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0f ? x[i] : a * x[i];
}
__global__ void k_lrelu_bwd(const float* dy, const float* x, float* dx, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = dy[i] * (x[i] > 0.0f ? 1.0f : a);
}

__global__ void k_elu_fwd(const float* x, float* y, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0f ? x[i] : a * (expf(x[i]) - 1.0f);
}
__global__ void k_elu_bwd(const float* dy, const float* x, float* dx, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = dy[i] * (x[i] > 0.0f ? 1.0f : a * expf(x[i]));
}

__global__ void k_sigmoid_fwd(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = sigmoidf_dev(x[i]);
}
__global__ void k_sigmoid_bwd(const float* dy, const float* y, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float s = y[i]; dx[i] = dy[i] * s * (1.0f - s); }
}

__global__ void k_tanh_fwd(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tanhf(x[i]);
}
__global__ void k_tanh_bwd(const float* dy, const float* y, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float t = y[i]; dx[i] = dy[i] * (1.0f - t * t); }
}

__global__ void k_gelu_fwd(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        float u  = GELU_C1 * (xi + GELU_C2 * xi * xi * xi);
        y[i] = 0.5f * xi * (1.0f + tanhf(u));
    }
}
__global__ void k_gelu_bwd(const float* dy, const float* x, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        float x2 = xi * xi;
        float u  = GELU_C1 * (xi + GELU_C2 * xi * x2);
        float t  = tanhf(u);
        float dudx = GELU_C1 * (1.0f + 3.0f * GELU_C2 * x2);
        float df   = 0.5f * (1.0f + t) + 0.5f * xi * (1.0f - t * t) * dudx;
        dx[i] = dy[i] * df;
    }
}

__global__ void k_silu_fwd(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * sigmoidf_dev(x[i]);
}
__global__ void k_silu_bwd(const float* dy, const float* x, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = sigmoidf_dev(x[i]);
        dx[i] = dy[i] * (s + x[i] * s * (1.0f - s));
    }
}

__global__ void k_softplus_fwd(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = softplusf_dev(x[i]);
}
__global__ void k_softplus_bwd(const float* dy, const float* x, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = dy[i] * sigmoidf_dev(x[i]);
}

__global__ void k_mish_fwd(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * tanhf(softplusf_dev(x[i]));
}
__global__ void k_mish_bwd(const float* dy, const float* x, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi  = x[i];
        float sp  = softplusf_dev(xi);
        float t   = tanhf(sp);
        float sig = sigmoidf_dev(xi);
        dx[i] = dy[i] * (t + xi * (1.0f - t * t) * sig);
    }
}

__global__ void k_hardtanh_fwd(const float* x, float* y, float lo, float hi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fminf(fmaxf(x[i], lo), hi);
}
__global__ void k_hardtanh_bwd(const float* dy, const float* x, float* dx,
                               float lo, float hi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = (x[i] > lo && x[i] < hi) ? dy[i] : 0.0f;
}

__global__ void k_hardsigmoid_fwd(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fminf(fmaxf((x[i] + 3.0f) / 6.0f, 0.0f), 1.0f);
}
__global__ void k_hardsigmoid_bwd(const float* dy, const float* x, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = (x[i] > -3.0f && x[i] < 3.0f) ? dy[i] * (1.0f / 6.0f) : 0.0f;
}

__global__ void k_hardswish_fwd(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        float h  = fminf(fmaxf((xi + 3.0f) / 6.0f, 0.0f), 1.0f);
        y[i] = xi * h;
    }
}
__global__ void k_hardswish_bwd(const float* dy, const float* x, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        float d;
        if      (xi <= -3.0f) d = 0.0f;
        else if (xi >=  3.0f) d = 1.0f;
        else                  d = xi / 3.0f + 0.5f;   // derivative of x*(x+3)/6
        dx[i] = dy[i] * d;
    }
}

// ---- Softmax: one thread block per row ------------------------------
__global__ void k_softmax_fwd(const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * cols;
    float*       yr = y + row * cols;

    extern __shared__ float smem[];  // size = blockDim.x
    int tid = threadIdx.x;

    // 1. max
    float m = -INFINITY;
    for (int i = tid; i < cols; i += blockDim.x) m = fmaxf(m, xr[i]);
    smem[tid] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float row_max = smem[0];

    // 2. sum of exp(x - max)
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float e = expf(xr[i] - row_max);
        yr[i] = e;                    // store exp in output for now
        sum  += e;
    }
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float row_sum = smem[0];
    float inv     = 1.0f / row_sum;

    // 3. normalize
    for (int i = tid; i < cols; i += blockDim.x) yr[i] *= inv;
}

// dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
__global__ void k_softmax_bwd(const float* dy, const float* y, float* dx,
                              int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* dyr = dy + row * cols;
    const float* yr  = y  + row * cols;
    float*       dxr = dx + row * cols;

    extern __shared__ float smem[];
    int tid = threadIdx.x;

    float dot = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) dot += dyr[i] * yr[i];
    smem[tid] = dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float row_dot = smem[0];

    for (int i = tid; i < cols; i += blockDim.x) {
        dxr[i] = yr[i] * (dyr[i] - row_dot);
    }
}

__global__ void k_log_softmax_fwd(const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * cols;
    float*       yr = y + row * cols;

    extern __shared__ float smem[];
    int tid = threadIdx.x;

    float m = -INFINITY;
    for (int i = tid; i < cols; i += blockDim.x) m = fmaxf(m, xr[i]);
    smem[tid] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float row_max = smem[0];

    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) sum += expf(xr[i] - row_max);
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float log_sum = logf(smem[0]) + row_max;

    for (int i = tid; i < cols; i += blockDim.x) yr[i] = xr[i] - log_sum;
}

// dL/dx_i = dL/dy_i - exp(y_i) * sum_j(dL/dy_j)
__global__ void k_log_softmax_bwd(const float* dy, const float* y, float* dx,
                                  int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* dyr = dy + row * cols;
    const float* yr  = y  + row * cols;
    float*       dxr = dx + row * cols;

    extern __shared__ float smem[];
    int tid = threadIdx.x;

    float s = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) s += dyr[i];
    smem[tid] = s;
    __syncthreads();
    for (int r = blockDim.x / 2; r > 0; r >>= 1) {
        if (tid < r) smem[tid] += smem[tid + r];
        __syncthreads();
    }
    float dy_sum = smem[0];

    for (int i = tid; i < cols; i += blockDim.x) {
        dxr[i] = dyr[i] - expf(yr[i]) * dy_sum;
    }
}

inline void launch_ew(const Tensor* t, int& blocks, int& threads) {
    threads = ULTRAML_CUDA_BLOCK;
    blocks  = (t->size + threads - 1) / threads;
}

// Choose threads-per-block for row-reduction kernels: power of two up to 512,
// not exceeding cols.
inline int rowwise_threads(int cols) {
    int t = 1;
    while (t < cols && t < 512) t <<= 1;
    return t < 32 ? 32 : t;
}

} // namespace

// ============ public wrappers ========================================
#define EW_UNARY(name, kf, kb, saved_arg)                                      \
    void name##_forward(const Tensor* input, Tensor* output) {                 \
        int b, t; launch_ew(input, b, t);                                      \
        kf<<<b, t>>>(input->data, output->data, input->size);                  \
        ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());                           \
    }                                                                          \
    void name##_backward(const Tensor* grad_output, const Tensor* saved_arg,   \
                         Tensor* grad_input) {                                 \
        int b, t; launch_ew(grad_output, b, t);                                \
        kb<<<b, t>>>(grad_output->data, saved_arg->data,                       \
                     grad_input->data, grad_output->size);                     \
        ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());                           \
    }

EW_UNARY(relu,     k_relu_fwd,     k_relu_bwd,     input)
EW_UNARY(sigmoid,  k_sigmoid_fwd,  k_sigmoid_bwd,  output)
EW_UNARY(tanh,     k_tanh_fwd,     k_tanh_bwd,     output)
EW_UNARY(gelu,     k_gelu_fwd,     k_gelu_bwd,     input)
EW_UNARY(silu,     k_silu_fwd,     k_silu_bwd,     input)
EW_UNARY(softplus, k_softplus_fwd, k_softplus_bwd, input)
EW_UNARY(mish,     k_mish_fwd,     k_mish_bwd,     input)
EW_UNARY(hardsigmoid, k_hardsigmoid_fwd, k_hardsigmoid_bwd, input)
EW_UNARY(hardswish,   k_hardswish_fwd,   k_hardswish_bwd,   input)

#undef EW_UNARY

// Param activations -------------------------------------------------------
void leaky_relu_forward(const Tensor* input, Tensor* output, float alpha) {
    int threads = ULTRAML_CUDA_BLOCK;
    int blocks  = (input->size + threads - 1) / threads;
    k_lrelu_fwd<<<blocks, threads>>>(input->data, output->data, alpha, input->size);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}
void leaky_relu_backward(const Tensor* grad_output, const Tensor* input,
                         Tensor* grad_input, float alpha) {
    int threads = ULTRAML_CUDA_BLOCK;
    int blocks  = (input->size + threads - 1) / threads;
    k_lrelu_bwd<<<blocks, threads>>>(grad_output->data, input->data,
                                     grad_input->data, alpha, input->size);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

void elu_forward(const Tensor* input, Tensor* output, float alpha) {
    int threads = ULTRAML_CUDA_BLOCK;
    int blocks  = (input->size + threads - 1) / threads;
    k_elu_fwd<<<blocks, threads>>>(input->data, output->data, alpha, input->size);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}
void elu_backward(const Tensor* grad_output, const Tensor* input,
                  Tensor* grad_input, float alpha) {
    int threads = ULTRAML_CUDA_BLOCK;
    int blocks  = (input->size + threads - 1) / threads;
    k_elu_bwd<<<blocks, threads>>>(grad_output->data, input->data,
                                   grad_input->data, alpha, input->size);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

void hardtanh_forward(const Tensor* input, Tensor* output,
                      float min_val, float max_val) {
    int threads = ULTRAML_CUDA_BLOCK;
    int blocks  = (input->size + threads - 1) / threads;
    k_hardtanh_fwd<<<blocks, threads>>>(input->data, output->data,
                                        min_val, max_val, input->size);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}
void hardtanh_backward(const Tensor* grad_output, const Tensor* input,
                       Tensor* grad_input, float min_val, float max_val) {
    int threads = ULTRAML_CUDA_BLOCK;
    int blocks  = (input->size + threads - 1) / threads;
    k_hardtanh_bwd<<<blocks, threads>>>(grad_output->data, input->data,
                                        grad_input->data, min_val, max_val,
                                        input->size);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

// Softmax / LogSoftmax ----------------------------------------------------
// Input must be 2D: [rows, cols]. softmax is applied along the last dim.
static void check_2d(const Tensor* t) {
    if (t->ndim != 2) {
        fprintf(stderr, "softmax/log_softmax requires 2D tensor, got ndim=%d\n",
                t->ndim);
        exit(EXIT_FAILURE);
    }
}

void softmax_forward(const Tensor* input, Tensor* output) {
    check_2d(input);
    int rows = input->shape[0];
    int cols = input->shape[1];
    int threads = rowwise_threads(cols);
    size_t smem = threads * sizeof(float);
    k_softmax_fwd<<<rows, threads, smem>>>(input->data, output->data, rows, cols);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

void softmax_backward(const Tensor* grad_output, const Tensor* output,
                      Tensor* grad_input) {
    check_2d(output);
    int rows = output->shape[0];
    int cols = output->shape[1];
    int threads = rowwise_threads(cols);
    size_t smem = threads * sizeof(float);
    k_softmax_bwd<<<rows, threads, smem>>>(grad_output->data, output->data,
                                           grad_input->data, rows, cols);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

void log_softmax_forward(const Tensor* input, Tensor* output) {
    check_2d(input);
    int rows = input->shape[0];
    int cols = input->shape[1];
    int threads = rowwise_threads(cols);
    size_t smem = threads * sizeof(float);
    k_log_softmax_fwd<<<rows, threads, smem>>>(input->data, output->data,
                                               rows, cols);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

void log_softmax_backward(const Tensor* grad_output, const Tensor* output,
                          Tensor* grad_input) {
    check_2d(output);
    int rows = output->shape[0];
    int cols = output->shape[1];
    int threads = rowwise_threads(cols);
    size_t smem = threads * sizeof(float);
    k_log_softmax_bwd<<<rows, threads, smem>>>(grad_output->data, output->data,
                                               grad_input->data, rows, cols);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace ultraml
