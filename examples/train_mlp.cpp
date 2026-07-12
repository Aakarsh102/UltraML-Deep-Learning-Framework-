// End-to-end training with the autograd engine: a small MLP on a synthetic
// 4-class problem. Compare with mlp.cpp, which wires every backward call by
// hand — here the tape records the forward pass and derives the backward.
//
// Per step:  forward (ops record nodes) -> loss -> autograd::backward
//            -> optimizer step -> zero_grad -> autograd::clear.

#include "../ultraml.h"

#include <cstdio>
#include <random>
#include <vector>

using namespace ultraml;

int main() {
    const int B = 256, IN = 16, HID = 64, CLASSES = 4, STEPS = 300;

    NNContext* ctx = create_context();
    nn::manual_seed(42);

    nn::Sequential model({
        new nn::Linear(IN, HID),
        new nn::LayerNorm(HID),
        new nn::GELU(),
        new nn::Dropout(0.1f),
        new nn::Linear(HID, CLASSES),
    });

    optim::AdamW opt(model.parameters(), 3e-3f);

    // Synthetic data: one random center per class, samples = center + noise.
    std::mt19937 gen(7);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::normal_distribution<float>       noise(0.0f, 0.35f);
    std::vector<float> centers(CLASSES * IN);
    for (float& c : centers) c = unit(gen);

    int x_shape[2] = { B, IN };
    Tensor* x = create_tensor(x_shape, 2);
    int* d_labels;
    ULTRAML_CUDA_CHECK(cudaMalloc(&d_labels, B * sizeof(int)));

    std::vector<float> hx(B * IN);
    std::vector<int>   hy(B);
    auto make_batch = [&]() {
        for (int i = 0; i < B; ++i) {
            int c = (int)(gen() % CLASSES);
            hy[i] = c;
            for (int j = 0; j < IN; ++j)
                hx[i * IN + j] = centers[c * IN + j] + noise(gen);
        }
        copy_from_host(x, hx.data());
        ULTRAML_CUDA_CHECK(cudaMemcpy(d_labels, hy.data(), B * sizeof(int),
                                      cudaMemcpyHostToDevice));
    };

    for (int step = 1; step <= STEPS; ++step) {
        make_batch();

        Tensor* logits = model.forward(ctx, x);
        Tensor* loss   = ops::cross_entropy(logits, d_labels, B, CLASSES);

        if (step == 1 || step % 50 == 0)
            std::printf("step %3d  loss %.4f\n", step, ops::item(loss));

        autograd::backward(ctx, loss);
        optim::clip_grad_norm(opt.params, 1.0f);
        opt.step();
        opt.zero_grad();
        autograd::clear();
    }

    // Evaluate on a fresh batch: eval mode (dropout off), no recording.
    model.eval();
    make_batch();
    {
        autograd::NoGradGuard no_grad;
        Tensor* logits = model.forward(ctx, x);
        std::vector<float> hl(B * CLASSES);
        copy_to_host(logits, hl.data());

        int correct = 0;
        for (int i = 0; i < B; ++i) {
            int best = 0;
            for (int c = 1; c < CLASSES; ++c)
                if (hl[i * CLASSES + c] > hl[i * CLASSES + best]) best = c;
            if (best == hy[i]) ++correct;
        }
        std::printf("eval accuracy: %.1f%%\n", 100.0 * correct / B);
    }
    autograd::clear();   // intermediates are tape-owned even under no-grad

    free_tensor(x);
    ULTRAML_CUDA_CHECK(cudaFree(d_labels));
    destroy_context(ctx);
    return 0;
}
