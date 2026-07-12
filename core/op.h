#pragma once

// Forward-declaration of the autograd graph node so Tensor can carry a
// grad_fn pointer without core/ depending on the autograd module. The full
// definition lives in autograd/autograd.h: a node holds the op name, the
// input tensors, the output, and a backward closure that captures whatever
// the forward pass saved.
//
// Ops themselves stay autograd-agnostic: every backward function in ultraml
// takes its required "saved" tensors as explicit parameters, and the
// recorded closures in autograd/ops.cu simply call those same functions.
// Code that only uses the low-level kernels never touches this type.

namespace ultraml {

struct AutogradNode;

} // namespace ultraml
