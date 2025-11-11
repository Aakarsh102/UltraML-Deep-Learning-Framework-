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

// ============ RELU ACTIVATION ============
__global__ void relu_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void relu_backward_kernel(const float* grad_output, const float* input,
                                     float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

void nn_relu_forward(Tensor* input, Tensor* output) {
    int threads = 256;
    int blocks = (input->size + threads - 1) / threads;
    relu_forward_kernel<<<blocks, threads>>>(input->data, output->data, input->size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void nn_relu_backward(Tensor* grad_output, Tensor* input, Tensor* grad_input) {
    int threads = 256;
    int blocks = (input->size + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(grad_output->data, input->data,
                                             grad_input->data, input->size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============ SIGMOID ACTIVATION ============
__global__ void sigmoid_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void sigmoid_backward_kernel(const float* grad_output, const float* output,
                                        float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float s = output[idx];
        grad_input[idx] = grad_output[idx] * s * (1.0f - s);
    }
}

void nn_sigmoid_forward(Tensor* input, Tensor* output) {
    int threads = 256;
    int blocks = (input->size + threads - 1) / threads;
    sigmoid_forward_kernel<<<blocks, threads>>>(input->data, output->data, input->size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void nn_sigmoid_backward(Tensor* grad_output, Tensor* output, Tensor* grad_input) {
    int threads = 256;
    int blocks = (output->size + threads - 1) / threads;
    sigmoid_backward_kernel<<<blocks, threads>>>(grad_output->data, output->data,
                                                grad_input->data, output->size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============ TANH ACTIVATION ============
__global__ void tanh_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void tanh_backward_kernel(const float* grad_output, const float* output,
                                     float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float t = output[idx];
        grad_input[idx] = grad_output[idx] * (1.0f - t * t);
    }
}

void nn_tanh_forward(Tensor* input, Tensor* output) {
    int threads = 256;
    int blocks = (input->size + threads - 1) / threads;
    tanh_forward_kernel<<<blocks, threads>>>(input->data, output->data, input->size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void nn_tanh_backward(Tensor* grad_output, Tensor* output, Tensor* grad_input) {
    int threads = 256;
    int blocks = (output->size + threads - 1) / threads;
    tanh_backward_kernel<<<blocks, threads>>>(grad_output->data, output->data,
                                             grad_input->data, output->size);
    CUDA_CHECK(cudaDeviceSynchronize());
}