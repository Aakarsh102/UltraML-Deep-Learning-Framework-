#pragma once

#include "../core/tensor.h"

// All activations are element-wise and shape-preserving. Each has a
// forward(input, output) and a backward that takes whichever saved tensor
// is cheapest: some save the input, others save the output. The saved
// tensor is named in the parameter list to make it obvious what a future
// autograd engine needs to stash during forward.
namespace ultraml {

// ---- ReLU: y = max(0, x) ----------------------------------------------
void relu_forward (const Tensor* input, Tensor* output);
void relu_backward(const Tensor* grad_output, const Tensor* input,
                   Tensor* grad_input);

// ---- LeakyReLU: y = x if x>0 else alpha*x -----------------------------
void leaky_relu_forward (const Tensor* input, Tensor* output, float alpha = 0.01f);
void leaky_relu_backward(const Tensor* grad_output, const Tensor* input,
                         Tensor* grad_input, float alpha = 0.01f);

// ---- ELU: y = x if x>0 else alpha*(exp(x)-1) --------------------------
void elu_forward (const Tensor* input, Tensor* output, float alpha = 1.0f);
void elu_backward(const Tensor* grad_output, const Tensor* input,
                  Tensor* grad_input, float alpha = 1.0f);

// ---- Sigmoid: y = 1/(1+exp(-x)) ---------------------------------------
void sigmoid_forward (const Tensor* input, Tensor* output);
void sigmoid_backward(const Tensor* grad_output, const Tensor* output,
                      Tensor* grad_input);

// ---- Tanh -------------------------------------------------------------
void tanh_forward (const Tensor* input, Tensor* output);
void tanh_backward(const Tensor* grad_output, const Tensor* output,
                   Tensor* grad_input);

// ---- GELU (tanh approximation) ----------------------------------------
void gelu_forward (const Tensor* input, Tensor* output);
void gelu_backward(const Tensor* grad_output, const Tensor* input,
                   Tensor* grad_input);

// ---- SiLU / Swish: y = x * sigmoid(x) ---------------------------------
void silu_forward (const Tensor* input, Tensor* output);
void silu_backward(const Tensor* grad_output, const Tensor* input,
                   Tensor* grad_input);

// ---- Softplus: y = log(1+exp(x)) --------------------------------------
void softplus_forward (const Tensor* input, Tensor* output);
void softplus_backward(const Tensor* grad_output, const Tensor* input,
                       Tensor* grad_input);

// ---- Mish: y = x * tanh(softplus(x)) ----------------------------------
void mish_forward (const Tensor* input, Tensor* output);
void mish_backward(const Tensor* grad_output, const Tensor* input,
                   Tensor* grad_input);

// ---- HardTanh: clamp(x, min, max) -------------------------------------
void hardtanh_forward (const Tensor* input, Tensor* output,
                       float min_val = -1.0f, float max_val = 1.0f);
void hardtanh_backward(const Tensor* grad_output, const Tensor* input,
                       Tensor* grad_input,
                       float min_val = -1.0f, float max_val = 1.0f);

// ---- HardSigmoid: clamp((x+3)/6, 0, 1) --------------------------------
void hardsigmoid_forward (const Tensor* input, Tensor* output);
void hardsigmoid_backward(const Tensor* grad_output, const Tensor* input,
                          Tensor* grad_input);

// ---- HardSwish: x * HardSigmoid(x) ------------------------------------
void hardswish_forward (const Tensor* input, Tensor* output);
void hardswish_backward(const Tensor* grad_output, const Tensor* input,
                        Tensor* grad_input);

// ---- Softmax (row-wise, numerically stable) ---------------------------
// For a 2D input [batch, features], softmax is applied along the last dim.
// Backward needs the output (saved tensor).
void softmax_forward (const Tensor* input, Tensor* output);
void softmax_backward(const Tensor* grad_output, const Tensor* output,
                      Tensor* grad_input);

// ---- LogSoftmax (row-wise, numerically stable) ------------------------
// Often paired with NLL loss. Backward needs the output.
void log_softmax_forward (const Tensor* input, Tensor* output);
void log_softmax_backward(const Tensor* grad_output, const Tensor* output,
                          Tensor* grad_input);

} // namespace ultraml
