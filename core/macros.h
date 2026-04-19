#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#define ULTRAML_CUDA_CHECK(call)                                               \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err_));             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define ULTRAML_CUBLAS_CHECK(call)                                             \
    do {                                                                       \
        cublasStatus_t st_ = (call);                                           \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error %s:%d (code %d)\n",                  \
                    __FILE__, __LINE__, (int)st_);                             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define ULTRAML_CUDNN_CHECK(call)                                              \
    do {                                                                       \
        cudnnStatus_t st_ = (call);                                            \
        if (st_ != CUDNN_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuDNN error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudnnGetErrorString(st_));             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define ULTRAML_CUDA_BLOCK 256
