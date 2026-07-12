# UltraML

A CUDA-backed deep-learning framework in C++. Layers, activations, norms,
and losses each come with forward + backward kernels, and a tape-based
autograd engine sits on top so full training loops need no hand-written
gradient code. The autograd layer is pluggable: any new op registers itself
with one call and immediately works with the module system and optimizers.

## Layout

```
ultraml.h                  umbrella header — one include to pull in everything
core/
    tensor.h/.cu           Tensor (data, shape, grad, grad_fn)
    context.h/.cu          NNContext (cuBLAS + cuDNN handles)
    macros.h               CUDA / cuBLAS / cuDNN error-check macros
    op.h                   forward-declares AutogradNode (defined in autograd/)
activations/
    activations.h/.cu      ReLU, LeakyReLU, ELU, Sigmoid, Tanh, GELU, SiLU,
                           Softplus, Mish, HardTanh, HardSigmoid, HardSwish,
                           Softmax, LogSoftmax
norms/
    norms.h                umbrella for all norm decls
    batchnorm.cu           BatchNorm1d, BatchNorm2d
    layernorm.cu           LayerNorm (over last axis)
    rmsnorm.cu             RMSNorm  (over last axis, no mean/beta)
    groupnorm.cu           GroupNorm
layers/
    layers.h               decls for everything below + element-wise ops
    linear.cu              cuBLAS-backed linear layer
    conv.cu                cuDNN-backed 2D convolution
    pool.cu                cuDNN-backed pooling
    matmul.cu              batched matmul, permute, causal attention mask
    dropout.cu             inverted-dropout mask generation
    embedding.cu           embedding gather / scatter-add
losses/
    losses.h/.cu           MSE, L1, Huber, CrossEntropy, BCE-with-logits
autograd/
    autograd.h/.cu         tape engine: record / backward / clear, NoGradGuard
    ops.h/.cu              differentiable ops:: wrappers over every kernel
nn/
    nn.h/.cu               Module, Sequential, Linear, Conv2d, pools, norms,
                           activations, Dropout, Embedding, Flatten,
                           MultiHeadAttention
    init.h/.cu             zeros/ones/uniform/normal/Xavier/Kaiming init
optim/
    optim.h/.cu            SGD (momentum/Nesterov), Adam, AdamW, grad clipping
examples/
    mlp.cpp                low-level API: every backward wired by hand
    train_mlp.cpp          autograd API: full training loop, ~20 lines of model
    transformer_block.cpp  causal attention block trained end-to-end
tools/
    host_check/check.sh    type-checks all sources with plain clang++ on
                           machines without the CUDA toolkit (stub headers)
CMakeLists.txt             builds libultraml and the examples
```

## Build

```
mkdir build && cd build
cmake ..
cmake --build . -j
./train_mlp_example
```

Override the CUDA arch if needed: `cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..`.

## Training with autograd

```cpp
#include "ultraml.h"
using namespace ultraml;

NNContext* ctx = create_context();

nn::Sequential model({
    new nn::Linear(784, 128),
    new nn::LayerNorm(128),
    new nn::GELU(),
    new nn::Dropout(0.1f),
    new nn::Linear(128, 10),
});
optim::AdamW opt(model.parameters(), 3e-4f);

for (...) {
    Tensor* logits = model.forward(ctx, x);                     // recorded
    Tensor* loss   = ops::cross_entropy(logits, labels, B, 10); // scalar [1]
    autograd::backward(ctx, loss);   // derive + accumulate all .grad
    opt.step();
    opt.zero_grad();
    autograd::clear();               // free the graph + intermediates
}

model.eval();                        // dropout off, BN uses running stats
{ autograd::NoGradGuard g; ... }     // inference without recording
```

Losses are ordinary graph tensors, so multi-task objectives compose:
`ops::add(l1, ops::scale(l2, 0.5f))` backpropagates both, correctly weighted.

## How autograd works (and how to extend it)

The engine (`autograd/autograd.h`) is a global tape of `AutogradNode`s. Each
node stores the op's inputs, its output, and a *closure* that maps the
output gradient to input gradients. During `autograd::backward(ctx, loss)`
the tape is walked in reverse topological order; gradients are accumulated
into `Tensor::grad`, so tensors consumed by several ops just work.

Making any op differentiable is one `record()` call after its forward:

```cpp
Tensor* my_op(Tensor* x) {
    Tensor* y = autograd::own(create_tensor(x->shape, x->ndim));
    my_op_forward(x, y);
    autograd::record("my_op", {x}, y,
        [x](NNContext*, const Tensor* gy, std::vector<Tensor*>& gi) {
            if (gi[0]) my_op_backward(gy, x, gi[0]);
        });
    return y;
}
```

The closure captures whatever the backward needs (saved tensors, scalars,
descriptors). `gi[i]` arrives zero-filled and shaped like input *i*, or
nullptr when that input doesn't need gradients; the engine accumulates it
into `inputs[i]->grad` afterwards. Every wrapper in `autograd/ops.cu`
follows this pattern — copy one as a template.

New *layers* go one level up: subclass `nn::Module`, allocate parameters
with `create_tensor(..., /*requires_grad=*/true)`, chain `ops::` calls in
`forward()`, and list parameters in `collect_parameters()`. No backward to
write — see `nn::MultiHeadAttention`, which is built entirely from recorded
primitives (linear, matmul, permute, softmax, dropout).

The low-level API is unchanged: every kernel can still be called directly
with hand-managed gradients (`examples/mlp.cpp`), and that path pays zero
overhead for the autograd machinery.

## Supported ops

| module       | ops                                                                 |
|--------------|---------------------------------------------------------------------|
| activations  | ReLU, LeakyReLU, ELU, Sigmoid, Tanh, GELU, SiLU, Softplus, Mish,    |
|              | HardTanh, HardSigmoid, HardSwish, Softmax, LogSoftmax               |
| norms        | BatchNorm1d, BatchNorm2d, LayerNorm, RMSNorm, GroupNorm             |
| layers       | Linear, Conv2d, MaxPool/AvgPool, batched MatMul, Dropout, Embedding |
| losses       | MSE, L1, Huber, CrossEntropy (logits + int labels), BCE-with-logits |
| nn           | Sequential + modules for all of the above, Flatten,                 |
|              | MultiHeadAttention (optional causal mask + attention dropout)       |
| optim        | SGD (momentum, Nesterov, weight decay), Adam, AdamW, clip_grad_norm |

All ops have forward + backward and a differentiable `ops::` wrapper.

## Notes & caveats

- The tape is a single global and is not thread-safe; run one graph at a
  time and call `autograd::clear()` between steps.
- Tensors returned by `ops::`/`nn::` calls are owned by the tape and freed
  by `autograd::clear()` — `copy_to_host` anything you need to keep.
- Don't mutate a tensor in-place after it has been used by a recorded op;
  wrappers allocate fresh outputs, so this only matters for user inputs.
- BatchNorm records gradients in training mode only (eval saves no batch
  statistics, matching the kernel contract).
- Losses differentiate wrt predictions; targets receive no gradient.
