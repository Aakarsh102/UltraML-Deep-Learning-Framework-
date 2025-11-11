#include "nn_ops.h"
#include <stdio.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t stat = call; \
        if (stat != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t stat = call; \
        if (stat != CUDNN_SUCCESS) { \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudnnGetErrorString(stat)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============ CONTEXT MANAGEMENT ============
NNContext* nn_create_context() {
    NNContext* ctx = (NNContext*)malloc(sizeof(NNContext));
    CUBLAS_CHECK(cublasCreate(&ctx->cublas_handle));
    CUDNN_CHECK(cudnnCreate(&ctx->cudnn_handle));
    return ctx;
}

void nn_destroy_context(NNContext* ctx) {
    CUBLAS_CHECK(cublasDestroy(ctx->cublas_handle));
    CUDNN_CHECK(cudnnDestroy(ctx->cudnn_handle));
    free(ctx);
}

// ============ TENSOR OPERATIONS ============
Tensor* nn_create_tensor(int* shape, int ndim) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->shape = (int*)malloc(ndim * sizeof(int));
    
    t->size = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }
    
    CUDA_CHECK(cudaMalloc(&t->data, t->size * sizeof(float)));
    return t;
}

void nn_free_tensor(Tensor* t) {
    CUDA_CHECK(cudaFree(t->data));
    free(t->shape);
    free(t);
}

__global__ void fill_kernel(float* data, int size, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

void nn_fill_tensor(Tensor* t, float value) {
    int threads = 256;
    int blocks = (t->size + threads - 1) / threads;
    fill_kernel<<<blocks, threads>>>(t->data, t->size, value);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void nn_copy_tensor(Tensor* dst, Tensor* src) {
    CUDA_CHECK(cudaMemcpy(dst->data, src->data, src->size * sizeof(float),
                         cudaMemcpyDeviceToDevice));
}


