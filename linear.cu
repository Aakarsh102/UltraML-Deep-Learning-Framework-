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


// ============ LINEAR LAYER ============
void nn_linear_forward(NNContext* ctx, 
                       Tensor* input,    // [batch, in_features]
                       Tensor* weight,   // [out_features, in_features]
                       Tensor* bias,     // [out_features]
                       Tensor* output)   // [batch, out_features]
{
    int batch = input->shape[0];
    int in_features = input->shape[1];
    int out_features = weight->shape[0];
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // output = input @ weight^T
    // C = alpha * A * B + beta * C
    // output[batch, out_features] = input[batch, in_features] @ weight^T[in_features, out_features]
    CUBLAS_CHECK(cublasSgemm(ctx->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            out_features, batch, in_features,
                            &alpha,
                            weight->data, in_features,
                            input->data, in_features,
                            &beta,
                            output->data, out_features));
    
    // Add bias (broadcast)
    if (bias != NULL) {
        __global__ void add_bias_kernel(float* output, const float* bias, 
                                       int batch, int features) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total = batch * features;
            if (idx < total) {
                int feat = idx % features;
                output[idx] += bias[feat];
            }
        }
        
        int total = batch * out_features;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        add_bias_kernel<<<blocks, threads>>>(output->data, bias->data, 
                                            batch, out_features);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void nn_linear_backward(NNContext* ctx,
                        Tensor* grad_output,  // [batch, out_features]
                        Tensor* input,        // [batch, in_features]
                        Tensor* weight,       // [out_features, in_features]
                        Tensor* grad_input,   // [batch, in_features]
                        Tensor* grad_weight,  // [out_features, in_features]
                        Tensor* grad_bias)    // [out_features]
{
    int batch = input->shape[0];
    int in_features = input->shape[1];
    int out_features = weight->shape[0];
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // grad_input = grad_output @ weight
    // grad_input[batch, in_features] = grad_output[batch, out_features] @ weight[out_features, in_features]
    CUBLAS_CHECK(cublasSgemm(ctx->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            in_features, batch, out_features,
                            &alpha,
                            weight->data, in_features,
                            grad_output->data, out_features,
                            &beta,
                            grad_input->data, in_features));
    
    // grad_weight = grad_output^T @ input
    // grad_weight[out_features, in_features] = grad_output^T[out_features, batch] @ input[batch, in_features]
    CUBLAS_CHECK(cublasSgemm(ctx->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            in_features, out_features, batch,
                            &alpha,
                            input->data, in_features,
                            grad_output->data, out_features,
                            &beta,
                            grad_weight->data, in_features));
    
    // grad_bias = sum(grad_output, dim=0)
    if (grad_bias != NULL) {
    
        int threads = 256;
        int blocks = (out_features + threads - 1) / threads;
        sum_grad_bias_kernel<<<blocks, threads>>>(grad_output->data, grad_bias->data,
                                                  batch, out_features);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}