#include "losses.h"
#include "../core/macros.h"

#include <cmath>

namespace ultraml {

namespace {

__device__ __forceinline__ float sigmoidf_dev(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template <typename Fn>
__global__ void reduce_kernel(const float* a, const float* b, float* out, int n, Fn f) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float v = 0.0f;
    if (idx < n) v = f(a[idx], b[idx]);
    smem[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, smem[0]);
}

// Host-side helper to run a reduction of element-wise f(pred[i], target[i]).
template <typename Fn>
float reduce_and_mean(const Tensor* a, const Tensor* b, Fn f) {
    float *d;
    ULTRAML_CUDA_CHECK(cudaMalloc(&d, sizeof(float)));
    ULTRAML_CUDA_CHECK(cudaMemset(d, 0, sizeof(float)));

    int t = 256;
    int blk = (a->size + t - 1) / t;
    size_t smem = t * sizeof(float);
    reduce_kernel<<<blk, t, smem>>>(a->data, b->data, d, a->size, f);

    float h; ULTRAML_CUDA_CHECK(cudaMemcpy(&h, d, sizeof(float), cudaMemcpyDeviceToHost));
    ULTRAML_CUDA_CHECK(cudaFree(d));
    return h / (float)a->size;
}

// ---- MSE ----------------------------------------------------------
struct MseFn { __device__ float operator()(float p, float t) const { float d=p-t; return d*d; } };
__global__ void mse_bwd_k(const float* p, const float* t, float* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = 2.0f * (p[i] - t[i]) / (float)n;
}

// ---- L1 -----------------------------------------------------------
struct L1Fn { __device__ float operator()(float p, float t) const { return fabsf(p-t); } };
__global__ void l1_bwd_k(const float* p, const float* t, float* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float d = p[i] - t[i];
        float s = d > 0 ? 1.0f : (d < 0 ? -1.0f : 0.0f);
        g[i] = s / (float)n;
    }
}

// ---- Huber --------------------------------------------------------
struct HuberFn {
    float delta;
    __device__ float operator()(float p, float t) const {
        float d = p - t;
        float a = fabsf(d);
        return (a < delta) ? 0.5f * d * d : delta * (a - 0.5f * delta);
    }
};
__global__ void huber_bwd_k(const float* p, const float* t, float* g,
                            int n, float delta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float d = p[i] - t[i];
        float a = fabsf(d);
        float grad = (a < delta) ? d : (d > 0 ? delta : -delta);
        g[i] = grad / (float)n;
    }
}

// ---- Cross-entropy over logits ------------------------------------
// One block per sample; threads cooperatively compute log-sum-exp of a row.
// Accumulates -(logit_target - lse) into d_loss.
__global__ void ce_fwd(const float* logits, const int* targets,
                       float* d_loss, int B, int C) {
    int b = blockIdx.x;
    if (b >= B) return;
    const float* row = logits + b * C;

    extern __shared__ float smem[];
    int tid = threadIdx.x;

    float m = -INFINITY;
    for (int j = tid; j < C; j += blockDim.x) m = fmaxf(m, row[j]);
    smem[tid] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid+s]);
        __syncthreads();
    }
    float row_max = smem[0];

    float sum = 0.0f;
    for (int j = tid; j < C; j += blockDim.x) sum += expf(row[j] - row_max);
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid+s];
        __syncthreads();
    }
    float lse = logf(smem[0]) + row_max;

    if (tid == 0) {
        float lt = row[targets[b]];
        atomicAdd(d_loss, lse - lt);
    }
}

__global__ void ce_bwd(const float* logits, const int* targets,
                       float* grad, int B, int C) {
    int b = blockIdx.x;
    if (b >= B) return;
    const float* row = logits + b * C;
    float*       gr  = grad   + b * C;

    extern __shared__ float smem[];
    int tid = threadIdx.x;

    float m = -INFINITY;
    for (int j = tid; j < C; j += blockDim.x) m = fmaxf(m, row[j]);
    smem[tid] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid+s]);
        __syncthreads();
    }
    float row_max = smem[0];

    float sum = 0.0f;
    for (int j = tid; j < C; j += blockDim.x) sum += expf(row[j] - row_max);
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid+s];
        __syncthreads();
    }
    float inv_sum = 1.0f / smem[0];
    int   tgt = targets[b];
    float inv_B = 1.0f / (float)B;

    for (int j = tid; j < C; j += blockDim.x) {
        float p = expf(row[j] - row_max) * inv_sum;
        gr[j] = (p - (j == tgt ? 1.0f : 0.0f)) * inv_B;
    }
}

// ---- BCE with logits ---------------------------------------------
// Stable BCE(logit x, target y) = max(x, 0) - x*y + log(1 + exp(-|x|))
struct BceFn {
    __device__ float operator()(float x, float y) const {
        float m = fmaxf(x, 0.0f);
        float ax = fabsf(x);
        return m - x * y + log1pf(expf(-ax));
    }
};
__global__ void bce_bwd_k(const float* x, const float* y, float* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = (sigmoidf_dev(x[i]) - y[i]) / (float)n;
}

inline int rowwise_threads(int cols) {
    int t = 1;
    while (t < cols && t < 512) t <<= 1;
    return t < 32 ? 32 : t;
}

} // namespace

float mse_loss(const Tensor* pred, const Tensor* target) {
    return reduce_and_mean(pred, target, MseFn());
}
void mse_loss_backward(const Tensor* pred, const Tensor* target, Tensor* grad_input) {
    int t = 256, blk = (pred->size + t - 1) / t;
    mse_bwd_k<<<blk, t>>>(pred->data, target->data, grad_input->data, pred->size);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

float l1_loss(const Tensor* pred, const Tensor* target) {
    return reduce_and_mean(pred, target, L1Fn());
}
void l1_loss_backward(const Tensor* pred, const Tensor* target, Tensor* grad_input) {
    int t = 256, blk = (pred->size + t - 1) / t;
    l1_bwd_k<<<blk, t>>>(pred->data, target->data, grad_input->data, pred->size);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

float huber_loss(const Tensor* pred, const Tensor* target, float delta) {
    return reduce_and_mean(pred, target, HuberFn{delta});
}
void huber_loss_backward(const Tensor* pred, const Tensor* target,
                         Tensor* grad_input, float delta) {
    int t = 256, blk = (pred->size + t - 1) / t;
    huber_bwd_k<<<blk, t>>>(pred->data, target->data, grad_input->data,
                            pred->size, delta);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

float cross_entropy_loss(const Tensor* logits, const int* targets_device,
                         int batch, int num_classes) {
    float* d; ULTRAML_CUDA_CHECK(cudaMalloc(&d, sizeof(float)));
    ULTRAML_CUDA_CHECK(cudaMemset(d, 0, sizeof(float)));
    int threads = rowwise_threads(num_classes);
    size_t smem = threads * sizeof(float);
    ce_fwd<<<batch, threads, smem>>>(logits->data, targets_device, d,
                                     batch, num_classes);
    float h; ULTRAML_CUDA_CHECK(cudaMemcpy(&h, d, sizeof(float), cudaMemcpyDeviceToHost));
    ULTRAML_CUDA_CHECK(cudaFree(d));
    return h / (float)batch;
}

void cross_entropy_backward(const Tensor* logits, const int* targets_device,
                            Tensor* grad_input, int batch, int num_classes) {
    int threads = rowwise_threads(num_classes);
    size_t smem = threads * sizeof(float);
    ce_bwd<<<batch, threads, smem>>>(logits->data, targets_device,
                                     grad_input->data, batch, num_classes);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

float bce_with_logits_loss(const Tensor* logits, const Tensor* target) {
    return reduce_and_mean(logits, target, BceFn());
}
void bce_with_logits_backward(const Tensor* logits, const Tensor* target,
                              Tensor* grad_input) {
    int t = 256, blk = (logits->size + t - 1) / t;
    bce_bwd_k<<<blk, t>>>(logits->data, target->data, grad_input->data, logits->size);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace ultraml
