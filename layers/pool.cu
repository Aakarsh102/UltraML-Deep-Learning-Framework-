#include "layers.h"
#include "../core/macros.h"

#include <cstdlib>

namespace ultraml {

PoolDescriptor* create_pool_descriptor(int batch, int channels,
                                       int in_h, int in_w,
                                       int window_h, int window_w,
                                       int stride_h, int stride_w,
                                       int pad_h, int pad_w,
                                       cudnnPoolingMode_t mode) {
    PoolDescriptor* desc = (PoolDescriptor*)std::malloc(sizeof(PoolDescriptor));

    ULTRAML_CUDNN_CHECK(cudnnCreatePoolingDescriptor(&desc->pooling_desc));
    ULTRAML_CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc->input_desc));
    ULTRAML_CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc->output_desc));

    ULTRAML_CUDNN_CHECK(cudnnSetPooling2dDescriptor(
        desc->pooling_desc, mode, CUDNN_PROPAGATE_NAN,
        window_h, window_w, pad_h, pad_w, stride_h, stride_w));

    ULTRAML_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        desc->input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, channels, in_h, in_w));

    int on, oc, oh, ow;
    ULTRAML_CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(
        desc->pooling_desc, desc->input_desc, &on, &oc, &oh, &ow));
    ULTRAML_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        desc->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        on, oc, oh, ow));
    return desc;
}

void free_pool_descriptor(PoolDescriptor* desc) {
    if (!desc) return;
    ULTRAML_CUDNN_CHECK(cudnnDestroyPoolingDescriptor(desc->pooling_desc));
    ULTRAML_CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc->input_desc));
    ULTRAML_CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc->output_desc));
    std::free(desc);
}

void pool2d_forward(NNContext* ctx, PoolDescriptor* desc,
                    const Tensor* input, Tensor* output) {
    const float alpha = 1.0f, beta = 0.0f;
    ULTRAML_CUDNN_CHECK(cudnnPoolingForward(
        ctx->cudnn_handle, desc->pooling_desc,
        &alpha, desc->input_desc,  input->data,
        &beta,  desc->output_desc, output->data));
}

void pool2d_backward(NNContext* ctx, PoolDescriptor* desc,
                     const Tensor* grad_output, const Tensor* output,
                     const Tensor* input, Tensor* grad_input) {
    const float alpha = 1.0f, beta = 0.0f;
    ULTRAML_CUDNN_CHECK(cudnnPoolingBackward(
        ctx->cudnn_handle, desc->pooling_desc,
        &alpha,
        desc->output_desc, output->data,
        desc->output_desc, grad_output->data,
        desc->input_desc,  input->data,
        &beta,
        desc->input_desc,  grad_input->data));
}

} // namespace ultraml
