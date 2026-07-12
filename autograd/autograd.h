#pragma once

// Tape-based (define-by-run) autograd engine.
//
// This module defines the AutogradNode that core/op.h forward-declares and
// that Tensor::grad_fn points to. It is deliberately generic: the engine
// knows nothing about specific ops. Any op — the built-in kernels or one you
// add later — becomes differentiable by calling autograd::record() after its
// forward pass with a closure that computes input gradients from the output
// gradient. Everything the backward needs (saved tensors, scalars, cuDNN
// descriptors, raw device pointers, ...) is captured inside that closure.
//
// Contract for the backward closure:
//   - grad_output is a shallow Tensor view whose ->data is d(loss)/d(output),
//     shaped like the op's output.
//   - grad_inputs[i] is a zero-filled scratch tensor shaped like inputs[i],
//     or nullptr when inputs[i] does not require grad (or was nullptr).
//     Write d(loss)/d(inputs[i]) into it; the engine then *accumulates* it
//     into inputs[i]->grad, so ops never worry about shared inputs.
//
// Ownership: tensors created inside op wrappers are registered with
// autograd::own() and freed by autograd::clear(). Leaf tensors (parameters,
// user inputs) are never owned by the tape. Typical training-step shape:
//
//   Tensor* loss = ops::...(...);         // forward, tape records nodes
//   autograd::backward(ctx, loss);        // reverse pass, fills .grad
//   optimizer.step();                     // consume parameter grads
//   autograd::clear();                    // free graph + intermediates
//
// The tape is a single global (not thread-safe); one graph at a time.

#include "../core/tensor.h"
#include "../core/context.h"

#include <functional>
#include <string>
#include <vector>

namespace ultraml {

// Computes gradients for one recorded op. See contract above.
using BackwardFn = std::function<void(NNContext* ctx,
                                      const Tensor* grad_output,
                                      std::vector<Tensor*>& grad_inputs)>;

struct AutogradNode {
    std::string          op_name;   // for debugging / error messages
    std::vector<Tensor*> inputs;    // may contain nullptr (e.g. absent bias)
    Tensor*              output;
    BackwardFn           backward_fn;
};

namespace autograd {

// ---- global recording switch ---------------------------------------
bool grad_enabled();
void set_grad_enabled(bool enabled);

// RAII guard: disables recording in its scope (inference / eval).
struct NoGradGuard {
    NoGradGuard();
    ~NoGradGuard();
  private:
    bool prev_;
};

// ---- tape -----------------------------------------------------------
// Registers a graph-owned tensor; freed by clear(). Returns t for chaining.
Tensor* own(Tensor* t);

// Records one op on the tape. No-op (and the node is not created) when
// grad is disabled or no input requires grad; otherwise sets
// output->grad_fn and output->requires_grad so downstream ops keep
// extending the graph.
void record(const char* op_name,
            std::vector<Tensor*> inputs,
            Tensor* output,
            BackwardFn backward_fn);

// Runs the reverse pass from `loss` (its grad is seeded with ones, so a
// scalar [1] loss gives standard gradients). Gradients are accumulated
// into every reachable tensor with requires_grad; call zero_grad() on
// parameters (or Optimizer::zero_grad) between steps.
void backward(NNContext* ctx, Tensor* loss);

// Frees all recorded nodes and every graph-owned tensor. Any Tensor*
// returned by an ops:: wrapper is invalid after this; copy values out
// (copy_to_host) first if you need them.
void clear();

// Number of nodes currently on the tape (introspection / tests).
size_t num_nodes();

} // namespace autograd
} // namespace ultraml
