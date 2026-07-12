#pragma once

// UltraML: a CUDA-backed deep-learning building-block library.
//
// Include this one header from C++ code to pull in every op. The library is
// split into semantic modules; each has its own header as well:
//
//   core/         Tensor, NNContext, CUDA error macros
//   activations/  ReLU, LeakyReLU, ELU, Sigmoid, Tanh, GELU, SiLU, Softplus,
//                 Mish, HardTanh, HardSigmoid, HardSwish, Softmax, LogSoftmax
//   norms/        BatchNorm1d, BatchNorm2d, LayerNorm, RMSNorm, GroupNorm
//   layers/       Linear, Conv2d, Pool2d, batched MatMul, Dropout,
//                 Embedding, element-wise tensor ops
//   losses/       MSE, L1, Huber, CrossEntropy, BCE-with-logits
//   autograd/     tape engine (autograd::record/backward/clear) and the
//                 differentiable ops:: wrappers over every kernel
//   nn/           Module system: Sequential, Linear, Conv2d, pools, norms,
//                 activations, Dropout, Embedding, MultiHeadAttention; init
//   optim/        SGD, Adam, AdamW, gradient clipping
//
// Two ways to use the library:
//
//   1. Low level — call *_forward / *_backward kernels yourself and manage
//      every gradient buffer by hand (see examples/mlp.cpp).
//   2. Autograd — chain ops:: / nn:: calls; the tape records each op and
//      autograd::backward() derives the whole reverse pass
//      (see examples/train_mlp.cpp and examples/transformer_block.cpp).
//
// Both layers share the same kernels; the ops:: wrappers in autograd/ops.h
// are thin recorded shims over the *_forward/*_backward pairs.

#include "core/macros.h"
#include "core/tensor.h"
#include "core/context.h"
#include "core/op.h"

#include "activations/activations.h"
#include "norms/norms.h"
#include "layers/layers.h"
#include "losses/losses.h"

#include "autograd/autograd.h"
#include "autograd/ops.h"
#include "nn/nn.h"
#include "nn/init.h"
#include "optim/optim.h"
