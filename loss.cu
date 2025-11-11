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


// ============ MSE LOSS ============
__global__ void mse_loss_kernel(const float* pred, const float* target, 
                               float* loss, int size) {
    __shared__ float shared_loss[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    float local_loss = 0.0f;
    if (idx < size) {
        float diff = pred[idx] - target[idx];
        local_loss = diff * diff;
    }
    shared_loss[tid] = local_loss;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_loss[tid] += shared_loss[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(loss, shared_loss[0]);
    }
}

__global__ void mse_backward_kernel(const float* pred, const float* target,
                                   float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 2.0f * (pred[idx] - target[idx]) / size;
    }
}

float nn_mse_loss(Tensor* pred, Tensor* target) {
    float* d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    
    int threads = 256;
    int blocks = (pred->size + threads - 1) / threads;
    mse_loss_kernel<<<blocks, threads>>>(pred->data, target->data, d_loss, pred->size);
    
    float h_loss;
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_loss));
    
    return h_loss / pred->size;
}

void nn_mse_loss_backward(Tensor* pred, Tensor* target, Tensor* grad_input) {
    int threads = 256;
    int blocks = (pred->size + threads - 1) / threads;
    mse_backward_kernel<<<blocks, threads>>>(pred->data, target->data,
                                            grad_input->data, pred->size);
    CUDA_CHECK(cudaDeviceSynchronize());
}