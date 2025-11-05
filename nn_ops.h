#ifndef NN_OPS_H
#define NN_OPS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

typedef struct {
    float *data;
    int* shape;
    int ndim;
    int size;
} Tensor

struct context_t {
    cublasHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;
};
typedef struct context_t NNContext;

NNContext* createNNContext();
void destroyNNContext(NNContext* context);

Tensor* nn_create_tensor(int* shape, int ndim);
void nn_free_tensor(Tensor* t);
void nn_fill_tensor(Tensor* t, float value);
void nn_copy_tensor(Tensor* dest, Tensor* src);



#endif // NN_OPS_H