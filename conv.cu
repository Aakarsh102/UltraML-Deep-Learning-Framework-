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


// ============ CONVOLUTION LAYER ============
ConvDescriptor* nn_create_conv_descriptor(NNContext* ctx,
                                          int batch, int in_channels, int in_h, int in_w,
                                          int out_channels, int kernel_h, int kernel_w,
                                          int stride_h, int stride_w,
                                          int pad_h, int pad_w)
{
    ConvDescriptor* desc = (ConvDescriptor*)malloc(sizeof(ConvDescriptor));
    
    // Create descriptors
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc->input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc->output_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc->filter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc->conv_desc));
    
    // Set input descriptor
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc->input_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          batch, in_channels, in_h, in_w));
    
    // Set filter descriptor
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(desc->filter_desc,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW,
                                          out_channels, in_channels, kernel_h, kernel_w));
    
    // Set convolution descriptor
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(desc->conv_desc,
                                               pad_h, pad_w,
                                               stride_h, stride_w,
                                               1, 1,
                                               CUDNN_CROSS_CORRELATION,
                                               CUDNN_DATA_FLOAT));
    
    // Get output dimensions
    int out_n, out_c, out_h, out_w;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(desc->conv_desc,
                                                      desc->input_desc,
                                                      desc->filter_desc,
                                                      &out_n, &out_c, &out_h, &out_w));
    
    // Set output descriptor
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc->output_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          out_n, out_c, out_h, out_w));
    
    // Get forward algorithm
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(ctx->cudnn_handle,
                                                    desc->input_desc,
                                                    desc->filter_desc,
                                                    desc->conv_desc,
                                                    desc->output_desc,
                                                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                    0,
                                                    &desc->fwd_algo));
    
    // Get workspace size
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(ctx->cudnn_handle,
                                                        desc->input_desc,
                                                        desc->filter_desc,
                                                        desc->conv_desc,
                                                        desc->output_desc,
                                                        desc->fwd_algo,
                                                        &desc->workspace_size));
    
    // Allocate workspace
    if (desc->workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&desc->workspace, desc->workspace_size));
    } else {
        desc->workspace = NULL;
    }
    
    // Get backward algorithms
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(ctx->cudnn_handle,
                                                         desc->filter_desc,
                                                         desc->output_desc,
                                                         desc->conv_desc,
                                                         desc->input_desc,
                                                         CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                         0,
                                                         &desc->bwd_data_algo));
    
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(ctx->cudnn_handle,
                                                           desc->input_desc,
                                                           desc->output_desc,
                                                           desc->conv_desc,
                                                           desc->filter_desc,
                                                           CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                           0,
                                                           &desc->bwd_filter_algo));
    
    return desc;
}

void nn_free_conv_descriptor(ConvDescriptor* desc) {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc->input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc->output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(desc->filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(desc->conv_desc));
    if (desc->workspace != NULL) {
        CUDA_CHECK(cudaFree(desc->workspace));
    }
    free(desc);
}

void nn_conv2d_forward(NNContext* ctx,
                       ConvDescriptor* desc,
                       Tensor* input,
                       Tensor* weight,
                       Tensor* bias,
                       Tensor* output)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    CUDNN_CHECK(cudnnConvolutionForward(ctx->cudnn_handle,
                                       &alpha,
                                       desc->input_desc, input->data,
                                       desc->filter_desc, weight->data,
                                       desc->conv_desc,
                                       desc->fwd_algo,
                                       desc->workspace, desc->workspace_size,
                                       &beta,
                                       desc->output_desc, output->data));
    
    if (bias != NULL) {
        cudnnTensorDescriptor_t bias_desc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              1, bias->shape[0], 1, 1));
        
        CUDNN_CHECK(cudnnAddTensor(ctx->cudnn_handle,
                                  &alpha,
                                  bias_desc, bias->data,
                                  &alpha,
                                  desc->output_desc, output->data));
        
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    }
}

void nn_conv2d_backward(NNContext* ctx,
                        ConvDescriptor* desc,
                        Tensor* grad_output,
                        Tensor* input,
                        Tensor* weight,
                        Tensor* grad_input,
                        Tensor* grad_weight,
                        Tensor* grad_bias)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Backward data
    CUDNN_CHECK(cudnnConvolutionBackwardData(ctx->cudnn_handle,
                                            &alpha,
                                            desc->filter_desc, weight->data,
                                            desc->output_desc, grad_output->data,
                                            desc->conv_desc,
                                            desc->bwd_data_algo,
                                            desc->workspace, desc->workspace_size,
                                            &beta,
                                            desc->input_desc, grad_input->data));
    
    // Backward filter
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(ctx->cudnn_handle,
                                              &alpha,
                                              desc->input_desc, input->data,
                                              desc->output_desc, grad_output->data,
                                              desc->conv_desc,
                                              desc->bwd_filter_algo,
                                              desc->workspace, desc->workspace_size,
                                              &beta,
                                              desc->filter_desc, grad_weight->data));
    
    // Backward bias
    if (grad_bias != NULL) {
        cudnnTensorDescriptor_t bias_desc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              1, grad_bias->shape[0], 1, 1));
        
        CUDNN_CHECK(cudnnConvolutionBackwardBias(ctx->cudnn_handle,
                                                &alpha,
                                                desc->output_desc, grad_output->data,
                                                &beta,
                                                bias_desc, grad_bias->data));
        
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    }
}