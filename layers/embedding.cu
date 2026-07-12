#include "layers.h"
#include "../core/macros.h"

namespace ultraml {

namespace {

__global__ void k_embed_fwd(const float* W, const int* ids, float* out,
                            int n, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * dim;
    if (i < total) {
        int row = i / dim;
        int d   = i % dim;
        out[i] = W[ids[row] * dim + d];
    }
}

// Rows may repeat in ids, so accumulation must be atomic.
__global__ void k_embed_bwd(const float* grad_out, const int* ids, float* gW,
                            int n, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * dim;
    if (i < total) {
        int row = i / dim;
        int d   = i % dim;
        atomicAdd(&gW[ids[row] * dim + d], grad_out[i]);
    }
}

} // namespace

void embedding_forward(const Tensor* weight, const int* indices_device,
                       int n_indices, Tensor* output) {
    int dim = weight->shape[weight->ndim - 1];
    int total = n_indices * dim;
    int t = ULTRAML_CUDA_BLOCK;
    int blk = (total + t - 1) / t;
    k_embed_fwd<<<blk, t>>>(weight->data, indices_device, output->data,
                            n_indices, dim);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

void embedding_backward(const Tensor* grad_output, const int* indices_device,
                        int n_indices, Tensor* grad_weight) {
    int dim = grad_weight->shape[grad_weight->ndim - 1];
    int total = n_indices * dim;
    int t = ULTRAML_CUDA_BLOCK;
    int blk = (total + t - 1) / t;
    k_embed_bwd<<<blk, t>>>(grad_output->data, indices_device,
                            grad_weight->data, n_indices, dim);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace ultraml
