#include "optim.h"
#include "../core/macros.h"

#include <cmath>
#include <utility>

namespace ultraml {
namespace optim {

namespace {

__global__ void k_sgd(float* p, const float* g, float* v,
                      float lr, float mu, float wd, bool nesterov, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float grad = g[i] + wd * p[i];
    if (v) {
        float vi = mu * v[i] + grad;
        v[i] = vi;
        grad = nesterov ? grad + mu * vi : vi;
    }
    p[i] -= lr * grad;
}

__global__ void k_adam(float* p, const float* g, float* m, float* v,
                       float lr, float b1, float b2, float eps, float wd,
                       float bias_c1, float bias_c2, bool decoupled, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float grad = g[i];
    if (!decoupled) grad += wd * p[i];

    float mi = b1 * m[i] + (1.0f - b1) * grad;
    float vi = b2 * v[i] + (1.0f - b2) * grad * grad;
    m[i] = mi;
    v[i] = vi;

    float m_hat = mi / bias_c1;
    float v_hat = vi / bias_c2;
    float update = m_hat / (sqrtf(v_hat) + eps);
    if (decoupled) update += wd * p[i];
    p[i] -= lr * update;
}

__global__ void k_sumsq(const float* g, float* out, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float x = (idx < n) ? g[idx] : 0.0f;
    smem[tid] = x * x;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, smem[0]);
}

__global__ void k_scale_inplace(float* g, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] *= s;
}

float* device_zeros(int n) {
    float* d;
    ULTRAML_CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    ULTRAML_CUDA_CHECK(cudaMemset(d, 0, n * sizeof(float)));
    return d;
}

inline void launch_dims(int n, int& blocks, int& threads) {
    threads = ULTRAML_CUDA_BLOCK;
    blocks  = (n + threads - 1) / threads;
}

} // namespace

// ==================== Optimizer ======================================

Optimizer::Optimizer(std::vector<Tensor*> parameters)
    : params(std::move(parameters)) {}

void Optimizer::zero_grad() {
    // Qualified: the member function name shadows the core helper.
    for (Tensor* p : params) ultraml::zero_grad(p);
}

// ==================== SGD ============================================

SGD::SGD(std::vector<Tensor*> parameters, float lr,
         float momentum, float weight_decay, bool nesterov)
    : Optimizer(std::move(parameters)),
      lr(lr), momentum(momentum), weight_decay(weight_decay),
      nesterov(nesterov) {
    if (momentum != 0.0f) {
        velocity_.reserve(params.size());
        for (Tensor* p : params) velocity_.push_back(device_zeros(p->size));
    }
}

SGD::~SGD() {
    for (float* v : velocity_) ULTRAML_CUDA_CHECK(cudaFree(v));
}

void SGD::step() {
    for (size_t i = 0; i < params.size(); ++i) {
        Tensor* p = params[i];
        if (!p->grad) continue;
        int blocks, threads;
        launch_dims(p->size, blocks, threads);
        k_sgd<<<blocks, threads>>>(p->data, p->grad,
                                   velocity_.empty() ? nullptr : velocity_[i],
                                   lr, momentum, weight_decay, nesterov,
                                   p->size);
    }
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

// ==================== Adam / AdamW ===================================

Adam::Adam(std::vector<Tensor*> parameters, float lr,
           float beta1, float beta2, float eps,
           float weight_decay, bool decoupled)
    : Optimizer(std::move(parameters)),
      lr(lr), beta1(beta1), beta2(beta2), eps(eps),
      weight_decay(weight_decay), decoupled(decoupled), t(0) {
    m_.reserve(params.size());
    v_.reserve(params.size());
    for (Tensor* p : params) {
        m_.push_back(device_zeros(p->size));
        v_.push_back(device_zeros(p->size));
    }
}

Adam::~Adam() {
    for (float* m : m_) ULTRAML_CUDA_CHECK(cudaFree(m));
    for (float* v : v_) ULTRAML_CUDA_CHECK(cudaFree(v));
}

void Adam::step() {
    ++t;
    float bias_c1 = 1.0f - std::pow(beta1, (float)t);
    float bias_c2 = 1.0f - std::pow(beta2, (float)t);
    for (size_t i = 0; i < params.size(); ++i) {
        Tensor* p = params[i];
        if (!p->grad) continue;
        int blocks, threads;
        launch_dims(p->size, blocks, threads);
        k_adam<<<blocks, threads>>>(p->data, p->grad, m_[i], v_[i],
                                    lr, beta1, beta2, eps, weight_decay,
                                    bias_c1, bias_c2, decoupled, p->size);
    }
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

// ==================== gradient clipping ==============================

float clip_grad_norm(const std::vector<Tensor*>& parameters, float max_norm) {
    float* d_sumsq = device_zeros(1);
    for (Tensor* p : parameters) {
        if (!p->grad) continue;
        int blocks, threads;
        launch_dims(p->size, blocks, threads);
        size_t smem = threads * sizeof(float);
        k_sumsq<<<blocks, threads, smem>>>(p->grad, d_sumsq, p->size);
    }
    float sumsq;
    ULTRAML_CUDA_CHECK(cudaMemcpy(&sumsq, d_sumsq, sizeof(float),
                                  cudaMemcpyDeviceToHost));
    ULTRAML_CUDA_CHECK(cudaFree(d_sumsq));

    float norm = std::sqrt(sumsq);
    if (norm > max_norm && norm > 0.0f) {
        float coef = max_norm / (norm + 1e-6f);
        for (Tensor* p : parameters) {
            if (!p->grad) continue;
            int blocks, threads;
            launch_dims(p->size, blocks, threads);
            k_scale_inplace<<<blocks, threads>>>(p->grad, coef, p->size);
        }
        ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
    }
    return norm;
}

} // namespace optim
} // namespace ultraml
