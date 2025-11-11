#ifndef NN_OPS_H
#define NN_OPS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

// Tensor structure
typedef struct {
    float* data;
    int* shape;
    int ndim;
    int size;
} Tensor;

// Context for cuBLAS and cuDNN
typedef struct {
    cublasHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;
} NNContext;

// Initialize/Destroy context
NNContext* nn_create_context();
void nn_destroy_context(NNContext* ctx);

// Tensor operations
Tensor* nn_create_tensor(int* shape, int ndim);
void nn_free_tensor(Tensor* t);
void nn_fill_tensor(Tensor* t, float value);
void nn_copy_tensor(Tensor* dst, Tensor* src);

// ============ LINEAR LAYER ============
// Forward: Y = X @ W^T + b
void nn_linear_forward(NNContext* ctx, 
                       Tensor* input,    // [batch, in_features]
                       Tensor* weight,   // [out_features, in_features]
                       Tensor* bias,     // [out_features]
                       Tensor* output);  // [batch, out_features]

// Backward: compute gradients
void nn_linear_backward(NNContext* ctx,
                        Tensor* grad_output,  // [batch, out_features]
                        Tensor* input,        // [batch, in_features]
                        Tensor* weight,       // [out_features, in_features]
                        Tensor* grad_input,   // [batch, in_features]
                        Tensor* grad_weight,  // [out_features, in_features]
                        Tensor* grad_bias);   // [out_features]

// ============ ACTIVATION FUNCTIONS ============
// ReLU
void nn_relu_forward(Tensor* input, Tensor* output);
void nn_relu_backward(Tensor* grad_output, Tensor* input, Tensor* grad_input);

// Sigmoid
void nn_sigmoid_forward(Tensor* input, Tensor* output);
void nn_sigmoid_backward(Tensor* grad_output, Tensor* output, Tensor* grad_input);

// Tanh
void nn_tanh_forward(Tensor* input, Tensor* output);
void nn_tanh_backward(Tensor* grad_output, Tensor* output, Tensor* grad_input);

// ============ CONVOLUTION LAYER ============
typedef struct {
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnConvolutionFwdAlgo_t fwd_algo;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
    void* workspace;
    size_t workspace_size;
} ConvDescriptor;

ConvDescriptor* nn_create_conv_descriptor(NNContext* ctx,
                                          int batch, int in_channels, int in_h, int in_w,
                                          int out_channels, int kernel_h, int kernel_w,
                                          int stride_h, int stride_w,
                                          int pad_h, int pad_w);
void nn_free_conv_descriptor(ConvDescriptor* desc);

void nn_conv2d_forward(NNContext* ctx,
                       ConvDescriptor* desc,
                       Tensor* input,    // [batch, in_channels, h, w]
                       Tensor* weight,   // [out_channels, in_channels, kh, kw]
                       Tensor* bias,     // [out_channels]
                       Tensor* output);  // [batch, out_channels, out_h, out_w]

void nn_conv2d_backward(NNContext* ctx,
                        ConvDescriptor* desc,
                        Tensor* grad_output,
                        Tensor* input,
                        Tensor* weight,
                        Tensor* grad_input,
                        Tensor* grad_weight,
                        Tensor* grad_bias);

// ============ POOLING ============
typedef struct {
    cudnnPoolingDescriptor_t pooling_desc;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
} PoolDescriptor;

PoolDescriptor* nn_create_pool_descriptor(int batch, int channels, int in_h, int in_w,
                                          int window_h, int window_w,
                                          int stride_h, int stride_w,
                                          int pad_h, int pad_w,
                                          cudnnPoolingMode_t mode);
void nn_free_pool_descriptor(PoolDescriptor* desc);

void nn_pool2d_forward(NNContext* ctx, PoolDescriptor* desc,
                       Tensor* input, Tensor* output);
void nn_pool2d_backward(NNContext* ctx, PoolDescriptor* desc,
                        Tensor* grad_output, Tensor* output, Tensor* input,
                        Tensor* grad_input);

// ============ BATCH NORMALIZATION ============
void nn_batchnorm_forward(Tensor* input,      // [batch, channels, h, w]
                          Tensor* gamma,      // [channels]
                          Tensor* beta,       // [channels]
                          Tensor* running_mean,  // [channels]
                          Tensor* running_var,   // [channels]
                          Tensor* output,
                          float momentum,
                          float eps,
                          bool training);

void nn_batchnorm_backward(Tensor* grad_output,
                           Tensor* input,
                           Tensor* gamma,
                           Tensor* mean,
                           Tensor* var,
                           Tensor* grad_input,
                           Tensor* grad_gamma,
                           Tensor* grad_beta,
                           float eps);

// ============ LOSS FUNCTIONS ============
// Mean Squared Error
float nn_mse_loss(Tensor* pred, Tensor* target);
void nn_mse_loss_backward(Tensor* pred, Tensor* target, Tensor* grad_input);

// Cross Entropy Loss (with softmax)
float nn_cross_entropy_loss(Tensor* logits, Tensor* targets);  // targets are class indices
void nn_cross_entropy_backward(Tensor* logits, Tensor* targets, Tensor* grad_input);

// ============ UTILITY FUNCTIONS ============
void nn_add(Tensor* a, Tensor* b, Tensor* output);
void nn_multiply(Tensor* a, Tensor* b, Tensor* output);
void nn_scale(Tensor* input, float scale, Tensor* output);
void nn_transpose(Tensor* input, Tensor* output);

#endif // NN_OPS_H