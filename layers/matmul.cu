#include "layers.h"
#include "../core/macros.h"

#include <cstdlib>

namespace ultraml {

namespace {

// Extract [batch, rows, cols] from a 2D or 3D tensor.
inline void bmm_dims(const Tensor* t, int& batch, int& rows, int& cols) {
    if (t->ndim == 3) {
        batch = t->shape[0]; rows = t->shape[1]; cols = t->shape[2];
    } else if (t->ndim == 2) {
        batch = 1; rows = t->shape[0]; cols = t->shape[1];
    } else {
        fprintf(stderr, "batched_matmul expects 2D or 3D tensors, got ndim=%d\n",
                t->ndim);
        exit(EXIT_FAILURE);
    }
}

__global__ void k_permute_0213(const float* in, float* out,
                               int d0, int d1, int d2, int d3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d0 * d1 * d2 * d3;
    if (i >= total) return;
    int i3 = i % d3; int t = i / d3;
    int i2 = t % d2; t /= d2;
    int i1 = t % d1;
    int i0 = t / d1;
    out[(((i0 * d2 + i2) * d1) + i1) * d3 + i3] = in[i];
}

__global__ void k_causal_mask(const float* in, float* out, int n, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int col = i % T;
        int row = (i / T) % T;
        out[i] = (col > row) ? in[i] - 1e9f : in[i];
    }
}

} // namespace

void batched_matmul(NNContext* ctx,
                    const Tensor* A, const Tensor* B, Tensor* C,
                    bool trans_a, bool trans_b) {
    int ab, ar, ac, bb, br, bc, cb, cr, cc;
    bmm_dims(A, ab, ar, ac);
    bmm_dims(B, bb, br, bc);
    bmm_dims(C, cb, cr, cc);

    int M  = trans_a ? ac : ar;
    int K  = trans_a ? ar : ac;
    int Kb = trans_b ? bc : br;
    int N  = trans_b ? br : bc;

    if (ab != bb || ab != cb || K != Kb || cr != M || cc != N) {
        fprintf(stderr,
                "batched_matmul shape mismatch: A[%d,%d,%d]%s B[%d,%d,%d]%s -> C[%d,%d,%d]\n",
                ab, ar, ac, trans_a ? "^T" : "",
                bb, br, bc, trans_b ? "^T" : "",
                cb, cr, cc);
        exit(EXIT_FAILURE);
    }

    const float alpha = 1.0f, beta = 0.0f;
    // row-major C = op(A) @ op(B)  ==  column-major C' = op(B)' * op(A)'
    // (same layout trick as linear.cu, per batch with fixed strides).
    ULTRAML_CUBLAS_CHECK(cublasSgemmStridedBatched(
        ctx->cublas_handle,
        trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
        trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B->data, bc, (long long)br * bc,
        A->data, ac, (long long)ar * ac,
        &beta,
        C->data, N,  (long long)M * N,
        ab));
}

void batched_matmul_backward(NNContext* ctx,
                             const Tensor* grad_output,
                             const Tensor* A, const Tensor* B,
                             bool trans_a, bool trans_b,
                             Tensor* grad_a, Tensor* grad_b) {
    // C = op(A) @ op(B), G = grad_output [batch, M, N].
    if (grad_a) {
        if (!trans_a) {
            // dA = G @ op(B)^T
            batched_matmul(ctx, grad_output, B, grad_a, false, !trans_b);
        } else {
            // A physical is [K, M]: dA = op(B) @ G^T
            batched_matmul(ctx, B, grad_output, grad_a, trans_b, true);
        }
    }
    if (grad_b) {
        if (!trans_b) {
            // dB = op(A)^T @ G
            batched_matmul(ctx, A, grad_output, grad_b, !trans_a, false);
        } else {
            // B physical is [N, K]: dB = G^T @ op(A)
            batched_matmul(ctx, grad_output, A, grad_b, true, trans_a);
        }
    }
}

void permute_0213(const Tensor* input, Tensor* output) {
    if (input->ndim != 4) {
        fprintf(stderr, "permute_0213 requires a 4D tensor, got ndim=%d\n",
                input->ndim);
        exit(EXIT_FAILURE);
    }
    int t = ULTRAML_CUDA_BLOCK;
    int blk = (input->size + t - 1) / t;
    k_permute_0213<<<blk, t>>>(input->data, output->data,
                               input->shape[0], input->shape[1],
                               input->shape[2], input->shape[3]);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

void add_causal_mask(const Tensor* input, Tensor* output, int seq_len) {
    int t = ULTRAML_CUDA_BLOCK;
    int blk = (input->size + t - 1) / t;
    k_causal_mask<<<blk, t>>>(input->data, output->data, input->size, seq_len);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace ultraml
