#include "tensor.h"
#include "macros.h"

#include <cstdlib>
#include <cstring>

namespace ultraml {

namespace {
__global__ void fill_kernel(float* data, int size, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = value;
}
} // namespace

Tensor* create_tensor(const int* shape, int ndim, bool requires_grad) {
    Tensor* t = (Tensor*)std::malloc(sizeof(Tensor));
    t->ndim  = ndim;
    t->shape = (int*)std::malloc(ndim * sizeof(int));
    t->size  = 1;
    for (int i = 0; i < ndim; ++i) {
        t->shape[i] = shape[i];
        t->size    *= shape[i];
    }

    ULTRAML_CUDA_CHECK(cudaMalloc(&t->data, t->size * sizeof(float)));

    t->grad          = nullptr;
    t->requires_grad = requires_grad;
    t->grad_fn       = nullptr;

    if (requires_grad) alloc_grad(t);
    return t;
}

void free_tensor(Tensor* t) {
    if (!t) return;
    if (t->grad) ULTRAML_CUDA_CHECK(cudaFree(t->grad));
    if (t->data) ULTRAML_CUDA_CHECK(cudaFree(t->data));
    std::free(t->shape);
    std::free(t);
}

void fill_tensor(Tensor* t, float value) {
    int threads = ULTRAML_CUDA_BLOCK;
    int blocks  = (t->size + threads - 1) / threads;
    fill_kernel<<<blocks, threads>>>(t->data, t->size, value);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

void zero_tensor(Tensor* t) {
    ULTRAML_CUDA_CHECK(cudaMemset(t->data, 0, t->size * sizeof(float)));
}

void copy_tensor(Tensor* dst, const Tensor* src) {
    ULTRAML_CUDA_CHECK(cudaMemcpy(dst->data, src->data,
                                  src->size * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
}

void copy_from_host(Tensor* dst, const float* host_data) {
    ULTRAML_CUDA_CHECK(cudaMemcpy(dst->data, host_data,
                                  dst->size * sizeof(float),
                                  cudaMemcpyHostToDevice));
}

void copy_to_host(const Tensor* src, float* host_data) {
    ULTRAML_CUDA_CHECK(cudaMemcpy(host_data, src->data,
                                  src->size * sizeof(float),
                                  cudaMemcpyDeviceToHost));
}

void alloc_grad(Tensor* t) {
    if (!t->grad) {
        ULTRAML_CUDA_CHECK(cudaMalloc(&t->grad, t->size * sizeof(float)));
    }
    ULTRAML_CUDA_CHECK(cudaMemset(t->grad, 0, t->size * sizeof(float)));
    t->requires_grad = true;
}

void zero_grad(Tensor* t) {
    if (t->grad) {
        ULTRAML_CUDA_CHECK(cudaMemset(t->grad, 0, t->size * sizeof(float)));
    }
}

void free_grad(Tensor* t) {
    if (t->grad) {
        ULTRAML_CUDA_CHECK(cudaFree(t->grad));
        t->grad = nullptr;
    }
}

} // namespace ultraml
