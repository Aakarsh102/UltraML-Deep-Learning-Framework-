// Host-side cuBLAS stub for syntax/type-checking only.
#pragma once

#include "cuda_runtime.h"

typedef enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1
} cublasStatus_t;

typedef struct cublasContext* cublasHandle_t;

typedef enum {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2
} cublasOperation_t;

cublasStatus_t cublasCreate(cublasHandle_t* handle);
cublasStatus_t cublasDestroy(cublasHandle_t handle);

cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float* alpha,
                           const float* A, int lda,
                           const float* B, int ldb,
                           const float* beta,
                           float* C, int ldc);

cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB,
    const float* beta,
    float* C, int ldc, long long int strideC,
    int batchCount);
