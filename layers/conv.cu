#include "layers.h"
#include "../core/macros.h"

#include <cstdlib>

namespace ultraml {

ConvDescriptor* create_conv_descriptor(NNContext* ctx,
                                       int batch, int in_channels,
                                       int in_h, int in_w,
                                       int out_channels,
                                       int kernel_h, int kernel_w,
                                       int stride_h, int stride_w,
                                       int pad_h, int pad_w) {
    ConvDescriptor* desc = (ConvDescriptor*)std::malloc(sizeof(ConvDescriptor));

    ULTRAML_CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc->input_desc));
    ULTRAML_CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc->output_desc));
    ULTRAML_CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc->filter_desc));
    ULTRAML_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc->conv_desc));

    ULTRAML_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        desc->input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, in_channels, in_h, in_w));

    ULTRAML_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        desc->filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        out_channels, in_channels, kernel_h, kernel_w));

    ULTRAML_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        desc->conv_desc,
        pad_h, pad_w, stride_h, stride_w, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    int on, oc, oh, ow;
    ULTRAML_CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
        desc->conv_desc, desc->input_desc, desc->filter_desc,
        &on, &oc, &oh, &ow));

    ULTRAML_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        desc->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        on, oc, oh, ow));

    // Pick algorithms via the v7 API (works across modern cuDNN).
    int ret_count;
    cudnnConvolutionFwdAlgoPerf_t fwd_perf;
    ULTRAML_CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
        ctx->cudnn_handle,
        desc->input_desc, desc->filter_desc, desc->conv_desc, desc->output_desc,
        1, &ret_count, &fwd_perf));
    desc->fwd_algo = fwd_perf.algo;

    cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perf;
    ULTRAML_CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        ctx->cudnn_handle,
        desc->filter_desc, desc->output_desc, desc->conv_desc, desc->input_desc,
        1, &ret_count, &bwd_data_perf));
    desc->bwd_data_algo = bwd_data_perf.algo;

    cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_perf;
    ULTRAML_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        ctx->cudnn_handle,
        desc->input_desc, desc->output_desc, desc->conv_desc, desc->filter_desc,
        1, &ret_count, &bwd_filter_perf));
    desc->bwd_filter_algo = bwd_filter_perf.algo;

    // Workspace: pick max over fwd and both bwd.
    size_t ws_fwd, ws_dx, ws_dw;
    ULTRAML_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        ctx->cudnn_handle,
        desc->input_desc, desc->filter_desc, desc->conv_desc, desc->output_desc,
        desc->fwd_algo, &ws_fwd));
    ULTRAML_CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        ctx->cudnn_handle,
        desc->filter_desc, desc->output_desc, desc->conv_desc, desc->input_desc,
        desc->bwd_data_algo, &ws_dx));
    ULTRAML_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        ctx->cudnn_handle,
        desc->input_desc, desc->output_desc, desc->conv_desc, desc->filter_desc,
        desc->bwd_filter_algo, &ws_dw));

    desc->workspace_size = ws_fwd;
    if (ws_dx > desc->workspace_size) desc->workspace_size = ws_dx;
    if (ws_dw > desc->workspace_size) desc->workspace_size = ws_dw;

    if (desc->workspace_size > 0) {
        ULTRAML_CUDA_CHECK(cudaMalloc(&desc->workspace, desc->workspace_size));
    } else {
        desc->workspace = nullptr;
    }
    return desc;
}

void free_conv_descriptor(ConvDescriptor* desc) {
    if (!desc) return;
    ULTRAML_CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc->input_desc));
    ULTRAML_CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc->output_desc));
    ULTRAML_CUDNN_CHECK(cudnnDestroyFilterDescriptor(desc->filter_desc));
    ULTRAML_CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(desc->conv_desc));
    if (desc->workspace) ULTRAML_CUDA_CHECK(cudaFree(desc->workspace));
    std::free(desc);
}

void conv2d_forward(NNContext* ctx, ConvDescriptor* desc,
                    const Tensor* input, const Tensor* weight, const Tensor* bias,
                    Tensor* output) {
    const float alpha = 1.0f, beta = 0.0f;

    ULTRAML_CUDNN_CHECK(cudnnConvolutionForward(
        ctx->cudnn_handle,
        &alpha,
        desc->input_desc,  input->data,
        desc->filter_desc, weight->data,
        desc->conv_desc, desc->fwd_algo,
        desc->workspace, desc->workspace_size,
        &beta,
        desc->output_desc, output->data));

    if (bias) {
        cudnnTensorDescriptor_t bias_desc;
        ULTRAML_CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        ULTRAML_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            1, bias->shape[0], 1, 1));
        ULTRAML_CUDNN_CHECK(cudnnAddTensor(
            ctx->cudnn_handle,
            &alpha, bias_desc, bias->data,
            &alpha, desc->output_desc, output->data));
        ULTRAML_CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    }
}

void conv2d_backward(NNContext* ctx, ConvDescriptor* desc,
                     const Tensor* grad_output,
                     const Tensor* input, const Tensor* weight,
                     Tensor* grad_input, Tensor* grad_weight, Tensor* grad_bias) {
    const float alpha = 1.0f, beta = 0.0f;

    if (grad_input) {
        ULTRAML_CUDNN_CHECK(cudnnConvolutionBackwardData(
            ctx->cudnn_handle,
            &alpha,
            desc->filter_desc, weight->data,
            desc->output_desc, grad_output->data,
            desc->conv_desc, desc->bwd_data_algo,
            desc->workspace, desc->workspace_size,
            &beta,
            desc->input_desc, grad_input->data));
    }

    if (grad_weight) {
        ULTRAML_CUDNN_CHECK(cudnnConvolutionBackwardFilter(
            ctx->cudnn_handle,
            &alpha,
            desc->input_desc, input->data,
            desc->output_desc, grad_output->data,
            desc->conv_desc, desc->bwd_filter_algo,
            desc->workspace, desc->workspace_size,
            &beta,
            desc->filter_desc, grad_weight->data));
    }

    if (grad_bias) {
        cudnnTensorDescriptor_t bias_desc;
        ULTRAML_CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        ULTRAML_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            1, grad_bias->shape[0], 1, 1));
        ULTRAML_CUDNN_CHECK(cudnnConvolutionBackwardBias(
            ctx->cudnn_handle,
            &alpha,
            desc->output_desc, grad_output->data,
            &beta,
            bias_desc, grad_bias->data));
        ULTRAML_CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    }
}

} // namespace ultraml
