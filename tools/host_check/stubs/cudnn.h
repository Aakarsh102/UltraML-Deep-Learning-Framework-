// Host-side cuDNN stub for syntax/type-checking only.
#pragma once

#include "cuda_runtime.h"

typedef enum {
    CUDNN_STATUS_SUCCESS = 0,
    CUDNN_STATUS_NOT_INITIALIZED = 1
} cudnnStatus_t;

typedef struct cudnnContext* cudnnHandle_t;
typedef struct cudnnTensorStruct* cudnnTensorDescriptor_t;
typedef struct cudnnFilterStruct* cudnnFilterDescriptor_t;
typedef struct cudnnConvolutionStruct* cudnnConvolutionDescriptor_t;
typedef struct cudnnPoolingStruct* cudnnPoolingDescriptor_t;

typedef enum { CUDNN_DATA_FLOAT = 0, CUDNN_DATA_DOUBLE = 1 } cudnnDataType_t;
typedef enum { CUDNN_TENSOR_NCHW = 0, CUDNN_TENSOR_NHWC = 1 } cudnnTensorFormat_t;
typedef enum { CUDNN_CONVOLUTION = 0, CUDNN_CROSS_CORRELATION = 1 } cudnnConvolutionMode_t;
typedef enum { CUDNN_NOT_PROPAGATE_NAN = 0, CUDNN_PROPAGATE_NAN = 1 } cudnnNanPropagation_t;

typedef enum {
    CUDNN_POOLING_MAX = 0,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2,
    CUDNN_POOLING_MAX_DETERMINISTIC = 3
} cudnnPoolingMode_t;

typedef enum { CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0 } cudnnConvolutionFwdAlgo_t;
typedef enum { CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0 } cudnnConvolutionBwdDataAlgo_t;
typedef enum { CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0 } cudnnConvolutionBwdFilterAlgo_t;

typedef struct {
    cudnnConvolutionFwdAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
} cudnnConvolutionFwdAlgoPerf_t;

typedef struct {
    cudnnConvolutionBwdDataAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
} cudnnConvolutionBwdDataAlgoPerf_t;

typedef struct {
    cudnnConvolutionBwdFilterAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
} cudnnConvolutionBwdFilterAlgoPerf_t;

const char* cudnnGetErrorString(cudnnStatus_t status);
cudnnStatus_t cudnnCreate(cudnnHandle_t* handle);
cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* d);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t d);
cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t d,
                                         cudnnTensorFormat_t format,
                                         cudnnDataType_t dataType,
                                         int n, int c, int h, int w);
cudnnStatus_t cudnnGetTensor4dDescriptor(cudnnTensorDescriptor_t d,
                                         cudnnDataType_t* dataType,
                                         int* n, int* c, int* h, int* w,
                                         int* nStride, int* cStride,
                                         int* hStride, int* wStride);

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* d);
cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t d);
cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t d,
                                         cudnnDataType_t dataType,
                                         cudnnTensorFormat_t format,
                                         int k, int c, int h, int w);

cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* d);
cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t d);
cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t d,
                                              int pad_h, int pad_w,
                                              int u, int v,
                                              int dilation_h, int dilation_w,
                                              cudnnConvolutionMode_t mode,
                                              cudnnDataType_t computeType);
cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t inputTensorDesc,
    cudnnFilterDescriptor_t filterDesc,
    int* n, int* c, int* h, int* w);

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t x, cudnnFilterDescriptor_t w,
    cudnnConvolutionDescriptor_t conv, cudnnTensorDescriptor_t y,
    int requestedAlgoCount, int* returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t* perfResults);
cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(
    cudnnHandle_t handle,
    cudnnFilterDescriptor_t w, cudnnTensorDescriptor_t dy,
    cudnnConvolutionDescriptor_t conv, cudnnTensorDescriptor_t dx,
    int requestedAlgoCount, int* returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t* perfResults);
cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t x, cudnnTensorDescriptor_t dy,
    cudnnConvolutionDescriptor_t conv, cudnnFilterDescriptor_t dw,
    int requestedAlgoCount, int* returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t* perfResults);

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t x, cudnnFilterDescriptor_t w,
    cudnnConvolutionDescriptor_t conv, cudnnTensorDescriptor_t y,
    cudnnConvolutionFwdAlgo_t algo, size_t* sizeInBytes);
cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t handle,
    cudnnFilterDescriptor_t w, cudnnTensorDescriptor_t dy,
    cudnnConvolutionDescriptor_t conv, cudnnTensorDescriptor_t dx,
    cudnnConvolutionBwdDataAlgo_t algo, size_t* sizeInBytes);
cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t x, cudnnTensorDescriptor_t dy,
    cudnnConvolutionDescriptor_t conv, cudnnFilterDescriptor_t dw,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t* sizeInBytes);

cudnnStatus_t cudnnConvolutionForward(
    cudnnHandle_t handle, const void* alpha,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void* workSpace, size_t workSpaceSizeInBytes,
    const void* beta, cudnnTensorDescriptor_t yDesc, void* y);
cudnnStatus_t cudnnAddTensor(
    cudnnHandle_t handle,
    const void* alpha, cudnnTensorDescriptor_t aDesc, const void* A,
    const void* beta, cudnnTensorDescriptor_t cDesc, void* C);
cudnnStatus_t cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void* alpha,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdDataAlgo_t algo,
    void* workSpace, size_t workSpaceSizeInBytes,
    const void* beta, cudnnTensorDescriptor_t dxDesc, void* dx);
cudnnStatus_t cudnnConvolutionBackwardFilter(
    cudnnHandle_t handle, const void* alpha,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdFilterAlgo_t algo,
    void* workSpace, size_t workSpaceSizeInBytes,
    const void* beta, cudnnFilterDescriptor_t dwDesc, void* dw);
cudnnStatus_t cudnnConvolutionBackwardBias(
    cudnnHandle_t handle, const void* alpha,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    const void* beta, cudnnTensorDescriptor_t dbDesc, void* db);

cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* d);
cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t d);
cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t d,
                                          cudnnPoolingMode_t mode,
                                          cudnnNanPropagation_t nan,
                                          int windowHeight, int windowWidth,
                                          int verticalPadding,
                                          int horizontalPadding,
                                          int verticalStride,
                                          int horizontalStride);
cudnnStatus_t cudnnGetPooling2dForwardOutputDim(
    cudnnPoolingDescriptor_t d, cudnnTensorDescriptor_t inputDesc,
    int* n, int* c, int* h, int* w);
cudnnStatus_t cudnnPoolingForward(
    cudnnHandle_t handle, cudnnPoolingDescriptor_t poolingDesc,
    const void* alpha, cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta, cudnnTensorDescriptor_t yDesc, void* y);
cudnnStatus_t cudnnPoolingBackward(
    cudnnHandle_t handle, cudnnPoolingDescriptor_t poolingDesc,
    const void* alpha,
    cudnnTensorDescriptor_t yDesc, const void* y,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta, cudnnTensorDescriptor_t dxDesc, void* dx);
