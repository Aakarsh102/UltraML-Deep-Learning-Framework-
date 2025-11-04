#include "nn_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t status = call;                                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                              \
            fprintf(stderr, "CUBLAS error in file '%s' in line %i.\n",      \
                    __FILE__, __LINE__);                                    \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

#define CUDNN_CHECK(call)                                                 \
    do {                                                                   \
        cudnnStatus_t status = call;                                       \
        if (status != CUDNN_STATUS_SUCCESS) {                              \
            fprintf(stderr, "CUDNN error in file '%s' in line %i : %s.\n",    \
                    __FILE__, __LINE__, cudnnGetErrorString(status));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)


// Using NNContext from nn_ops.h that has cudNNHandle_t and CUBLASHandle_t

NNContext* createNNContext() {
    NNContext *context = (NNContext*)malloc(sizeof(NNContext));
    if (context == NULL) {
        fprintf(stderr, "error allocating NNContext\n");
        exit(EXIT_FAILURE);
    }
    CUDNN_CHECK(cudnnCreate(&context -> cudnnHandle));
    CUBLAS_CHECK(cublasCreate(&context -> cublasHandle));
    return context;
}

void destroyNNContext(NNContext* context) {
    if (context != NULL) {
        CUDNN_CHECK(cudnnDestroy(context -> cudnnHandle));
        CUBLAS_CHECK(cublasDestroy(context -> cublasHandle));
        free(context);
    }
}

