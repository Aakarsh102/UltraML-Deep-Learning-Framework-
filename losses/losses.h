#pragma once

#include "../core/tensor.h"

// Losses return a single scalar (mean over all elements unless noted).
// Each has a corresponding backward that writes d(loss)/d(pred) into
// grad_input, shaped like pred. Reduction is "mean" for gradient scaling.
namespace ultraml {

// ---- Mean Squared Error ----------------------------------------------
float mse_loss(const Tensor* pred, const Tensor* target);
void  mse_loss_backward(const Tensor* pred, const Tensor* target,
                        Tensor* grad_input);

// ---- L1 (Mean Absolute Error) ----------------------------------------
float l1_loss(const Tensor* pred, const Tensor* target);
void  l1_loss_backward(const Tensor* pred, const Tensor* target,
                       Tensor* grad_input);

// ---- Huber / Smooth L1 -----------------------------------------------
// |x| < delta: 0.5 * x^2 ; else: delta * (|x| - 0.5 * delta)
float huber_loss(const Tensor* pred, const Tensor* target, float delta = 1.0f);
void  huber_loss_backward(const Tensor* pred, const Tensor* target,
                          Tensor* grad_input, float delta = 1.0f);

// ---- Cross-entropy over logits ---------------------------------------
// logits: [batch, num_classes]. targets: int32 class indices, [batch].
// Returns mean loss over batch; backward writes grad wrt logits.
float cross_entropy_loss(const Tensor* logits, const int* targets_device,
                         int batch, int num_classes);
void  cross_entropy_backward(const Tensor* logits, const int* targets_device,
                             Tensor* grad_input, int batch, int num_classes);

// ---- Binary cross-entropy with logits --------------------------------
// Numerically-stable version: BCE(sigmoid(x), y) expressed directly on x.
// pred and target have identical shape. Mean reduction.
float bce_with_logits_loss(const Tensor* logits, const Tensor* target);
void  bce_with_logits_backward(const Tensor* logits, const Tensor* target,
                               Tensor* grad_input);

} // namespace ultraml
