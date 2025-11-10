#include "nn_ops.h"

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

__global__ void add_bias_kernel(float *input, float *bias, int total_elements, int dim) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_elements) {
        int feature = idx % dim;
        input[idx] = bias[feature];
    }
}

void nn_linear_forward(NNContext* context,
                           const Tensor* input, 
                           const Tensor* weight,
                           const Tensor* bias,
                           Tensor* output
                    ) {

        int batch_size = input->shape[0];
        int input_dim = input->shape[1];
        int output_dim = weight->shape[0];
        // I'll be doing output = x * W^T + b

        const float alpha = 1.0f;
        const float beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(
            context -> cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            output_dim, batch_size, input_dim, &alpha,
            weight -> data, input_dim, input -> data, batch_size, &beta,
            output -> data, output_dim 
        ));
        if (bias != nullptr) {
            int total_elems = batch_size * output_dim;
            int threadsPerBlock = 256;
            int blocksPerGrid = (total_elems + threadsPerBlock - 1) / threadsPerBlock;
            add_bias_kernel<<<blocksPerGrid, threadsPerBlock>>> (output -> data, bias -> data, total_elems, output_dim);
        }
        



    }