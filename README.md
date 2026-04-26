# UltraML

A CUDA-backed building-block library for deep learning in C++. Layers,
activations, norms, and losses each come with forward + backward kernels.
The API is designed so that a future autograd graph can be layered on top
without reimplementing anything.

## Layout

```
ultraml.h                  umbrella header — one include to pull in everything
core/
    tensor.h/.cu           Tensor (data, shape, optional grad + grad_fn)
    context.h/.cu          NNContext (cuBLAS + cuDNN handles)
    macros.h               CUDA / cuBLAS / cuDNN error-check macros
    op.h                   forward-declares AutogradNode (future autograd)
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
    layers.h               Linear, Conv2d, Pool2d, element-wise ops
    linear.cu              cuBLAS-backed linear layer
    conv.cu                cuDNN-backed 2D convolution
    pool.cu                cuDNN-backed pooling
losses/
    losses.h/.cu           MSE, L1, Huber, CrossEntropy, BCE-with-logits
examples/
    mlp.cpp                2-layer MLP forward + backward
CMakeLists.txt             builds libultraml and the example
```

## Build

```
mkdir build && cd build
cmake ..
cmake --build . -j
./mlp_example
```

Override the CUDA arch if needed: `cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..`.

## Using it

Every op lives in `namespace ultraml`. Allocate `Tensor*` for activations,
parameters, and gradients, then chain calls:

```cpp
#include "ultraml.h"
using namespace ultraml;

NNContext* ctx = create_context();

int shape_x[2]  = {B, IN};
int shape_w[2]  = {OUT, IN};
Tensor* x      = create_tensor(shape_x, 2);
Tensor* W      = create_tensor(shape_w, 2);
Tensor* y      = create_tensor((int[]){B, OUT}, 2);

linear_forward(ctx, x, W, /*bias=*/nullptr, y);
relu_forward(y, y);

// ... backward similarly; see examples/mlp.cpp
```

## Autograd-ready design

There is no autograd graph yet, but the pieces are in place:

- `Tensor` carries `grad`, `requires_grad`, and an opaque `grad_fn*`. Call
  `alloc_grad(t)` to allocate the gradient buffer.
- `AutogradNode` is forward-declared in `core/op.h`. When the autograd module
  lands, it will hold `(inputs, backward_fn, saved_tensors)` and hook into
  `Tensor::grad_fn` during forward passes.
- Every backward already takes its "saved" tensors as explicit arguments
  (e.g. LayerNorm backward takes `saved_mean`/`saved_inv_std`). The same
  tensors that the caller currently stashes manually are what autograd will
  record automatically — no kernel changes required.

So: add an `autograd/` module later, call the existing forward/backward
functions from its generated nodes, and user code written against the
low-level API continues to work unchanged.

## Supported ops

| module       | ops                                                                 |
|--------------|---------------------------------------------------------------------|
| activations  | ReLU, LeakyReLU, ELU, Sigmoid, Tanh, GELU, SiLU, Softplus, Mish,    |
|              | HardTanh, HardSigmoid, HardSwish, Softmax, LogSoftmax               |
| norms        | BatchNorm1d, BatchNorm2d, LayerNorm, RMSNorm, GroupNorm             |
| layers       | Linear, Conv2d, MaxPool/AvgPool (via cuDNN mode)                    |
| losses       | MSE, L1, Huber, CrossEntropy (logits + int labels), BCE-with-logits |

All ops have forward + backward.
