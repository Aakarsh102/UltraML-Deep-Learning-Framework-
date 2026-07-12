#pragma once

// Optimizers. Each consumes the .grad buffers that autograd::backward()
// (or a hand-written backward pass) filled on the parameter tensors.
// Typical loop:
//
//   optim::AdamW opt(model.parameters(), 3e-4f);
//   ...forward...; autograd::backward(ctx, loss);
//   opt.step();
//   opt.zero_grad();
//   autograd::clear();
//
// Optimizer state (momentum / moment buffers) lives on the device, one
// buffer per parameter, allocated in the constructor and freed with the
// optimizer. Parameters whose grad is nullptr are skipped by step().

#include "../core/tensor.h"

#include <vector>

namespace ultraml {
namespace optim {

struct Optimizer {
    explicit Optimizer(std::vector<Tensor*> parameters);
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    void zero_grad();               // zero every parameter's grad buffer

    std::vector<Tensor*> params;
};

// SGD with optional momentum, Nesterov, and (coupled) L2 weight decay.
struct SGD : Optimizer {
    SGD(std::vector<Tensor*> parameters, float lr,
        float momentum = 0.0f, float weight_decay = 0.0f,
        bool nesterov = false);
    ~SGD() override;
    void step() override;

    float lr, momentum, weight_decay;
    bool nesterov;

  private:
    std::vector<float*> velocity_;   // empty when momentum == 0
};

// Adam. With decoupled = true this is AdamW (decay applied to the weights
// directly instead of being added to the gradient).
struct Adam : Optimizer {
    Adam(std::vector<Tensor*> parameters, float lr = 1e-3f,
         float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
         float weight_decay = 0.0f, bool decoupled = false);
    ~Adam() override;
    void step() override;

    float lr, beta1, beta2, eps, weight_decay;
    bool decoupled;
    int t;   // step count (for bias correction)

  private:
    std::vector<float*> m_, v_;
};

struct AdamW : Adam {
    explicit AdamW(std::vector<Tensor*> parameters, float lr = 1e-3f,
                   float beta1 = 0.9f, float beta2 = 0.999f,
                   float eps = 1e-8f, float weight_decay = 0.01f)
        : Adam(std::move(parameters), lr, beta1, beta2, eps,
               weight_decay, /*decoupled=*/true) {}
};

// Rescales all gradients so their global L2 norm is at most max_norm.
// Returns the norm measured before clipping.
float clip_grad_norm(const std::vector<Tensor*>& parameters, float max_norm);

} // namespace optim
} // namespace ultraml
