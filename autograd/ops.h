#pragma once

// Differentiable functional wrappers over the raw kernels.
//
// Every function here (1) allocates a fresh graph-owned output, (2) runs the
// existing forward kernel, and (3) records a node on the autograd tape whose
// closure calls the existing backward kernel with the tensors it saved.
// Chain them freely; then autograd::backward(ctx, loss) fills .grad on every
// tensor with requires_grad, and autograd::clear() frees the intermediates.
//
// This is also the extension point: a new op only needs a forward kernel, a
// backward kernel, and a ~10-line wrapper in this style to plug into the
// graph — the engine, nn:: modules, and optimizers all pick it up for free.
//
// Notes:
//   - inputs are non-const because backward writes into their ->grad.
//   - losses differentiate wrt predictions only (targets get no grad).
//   - batchnorm records only in training mode (eval saves no statistics).

#include "../core/tensor.h"
#include "../core/context.h"
#include "../layers/layers.h"

#include <initializer_list>

namespace ultraml {
namespace ops {

// ---- activations (shape-preserving) ----------------------------------
Tensor* relu       (Tensor* x);
Tensor* leaky_relu (Tensor* x, float alpha = 0.01f);
Tensor* elu        (Tensor* x, float alpha = 1.0f);
Tensor* sigmoid    (Tensor* x);
Tensor* tanh       (Tensor* x);
Tensor* gelu       (Tensor* x);
Tensor* silu       (Tensor* x);
Tensor* softplus   (Tensor* x);
Tensor* mish       (Tensor* x);
Tensor* hardtanh   (Tensor* x, float min_val = -1.0f, float max_val = 1.0f);
Tensor* hardsigmoid(Tensor* x);
Tensor* hardswish  (Tensor* x);
Tensor* softmax    (Tensor* x);   // 2D [rows, cols], along last dim
Tensor* log_softmax(Tensor* x);   // 2D [rows, cols], along last dim

// ---- layers -----------------------------------------------------------
// y = x @ W^T + b. x [batch, in], W [out, in], b [out] or nullptr.
Tensor* linear(NNContext* ctx, Tensor* x, Tensor* W, Tensor* b);

// Output shape is read from the descriptor. b may be nullptr.
Tensor* conv2d(NNContext* ctx, ConvDescriptor* desc,
               Tensor* x, Tensor* W, Tensor* b);

Tensor* pool2d(NNContext* ctx, PoolDescriptor* desc, Tensor* x);

// ---- norms --------------------------------------------------------------
Tensor* layernorm(Tensor* x, Tensor* gamma, Tensor* beta, float eps = 1e-5f);
Tensor* rmsnorm  (Tensor* x, Tensor* gamma, float eps = 1e-5f);
Tensor* groupnorm(Tensor* x, Tensor* gamma, Tensor* beta, int num_groups,
                  float eps = 1e-5f);
Tensor* batchnorm1d(Tensor* x, Tensor* gamma, Tensor* beta,
                    Tensor* running_mean, Tensor* running_var,
                    float momentum, float eps, bool training);
Tensor* batchnorm2d(Tensor* x, Tensor* gamma, Tensor* beta,
                    Tensor* running_mean, Tensor* running_var,
                    float momentum, float eps, bool training);

// ---- element-wise / shape ------------------------------------------------
Tensor* add  (Tensor* a, Tensor* b);          // same shape
Tensor* mul  (Tensor* a, Tensor* b);          // same shape
Tensor* scale(Tensor* x, float s);
Tensor* reshape(Tensor* x, const int* shape, int ndim);   // size must match
Tensor* reshape(Tensor* x, std::initializer_list<int> shape);

// ---- new functionality -----------------------------------------------
// Inverted dropout; identity when !training or p <= 0.
Tensor* dropout(Tensor* x, float p, bool training);

// weight [vocab, dim], indices_device: n int32 ids on device -> [n, dim].
Tensor* embedding(Tensor* weight, const int* indices_device, int n_indices);

// C = op(a) @ op(b); 2D or 3D-batched (see batched_matmul in layers.h).
Tensor* matmul(NNContext* ctx, Tensor* a, Tensor* b,
               bool trans_a = false, bool trans_b = false);

// Additive causal mask over trailing [T, T] score tiles.
Tensor* causal_mask(Tensor* scores, int seq_len);

// [d0, d1, d2, d3] -> [d0, d2, d1, d3]; backward is the same permute.
Tensor* permute0213(Tensor* x);

// ---- losses (return graph-owned scalar [1] tensors) --------------------
Tensor* mse_loss     (Tensor* pred, Tensor* target);
Tensor* l1_loss      (Tensor* pred, Tensor* target);
Tensor* huber_loss   (Tensor* pred, Tensor* target, float delta = 1.0f);
Tensor* cross_entropy(Tensor* logits, const int* targets_device,
                      int batch, int num_classes);
Tensor* bce_with_logits(Tensor* logits, Tensor* target);

// Convenience: copy a scalar [1] loss tensor's value to the host.
float item(const Tensor* scalar);

} // namespace ops
} // namespace ultraml
