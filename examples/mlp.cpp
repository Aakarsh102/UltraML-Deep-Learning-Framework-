// Example: a 2-layer MLP doing one forward + backward pass on random data.
//
// Network: x [B, 784] -> Linear -> LayerNorm -> GELU -> Linear -> logits [B, 10]
// Loss:    cross-entropy on random integer labels
//
// The point is to show how the pieces compose; it is not a training loop.
// Plug in an optimizer (SGD / Adam) over the dW*/db* tensors to complete one.

#include "../ultraml.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <vector>

using namespace ultraml;

static Tensor* mktensor(std::initializer_list<int> shape) {
    std::vector<int> s(shape);
    return create_tensor(s.data(), (int)s.size(), false);
}

static void randomize(Tensor* t, float scale) {
    std::vector<float> host(t->size);
    for (int i = 0; i < t->size; ++i) {
        host[i] = scale * ((float)std::rand() / RAND_MAX - 0.5f);
    }
    copy_from_host(t, host.data());
}

int main() {
    const int B = 32, IN = 784, HID = 128, OUT = 10;
    NNContext* ctx = create_context();

    // Parameters
    Tensor* W1 = mktensor({HID, IN});  randomize(W1, 0.1f);
    Tensor* b1 = mktensor({HID});      zero_tensor(b1);
    Tensor* W2 = mktensor({OUT, HID}); randomize(W2, 0.1f);
    Tensor* b2 = mktensor({OUT});      zero_tensor(b2);

    // LayerNorm params on the HID dim
    Tensor* ln_g = mktensor({HID}); fill_tensor(ln_g, 1.0f);
    Tensor* ln_b = mktensor({HID}); zero_tensor(ln_b);

    // Intermediates needed by backward
    Tensor* x      = mktensor({B, IN});  randomize(x, 1.0f);
    Tensor* h1     = mktensor({B, HID});
    Tensor* h1_ln  = mktensor({B, HID});
    Tensor* h1_a   = mktensor({B, HID});
    Tensor* logits = mktensor({B, OUT});

    // Saved state for LayerNorm backward
    Tensor* ln_mean = mktensor({B});
    Tensor* ln_istd = mktensor({B});

    // Random integer labels on device
    std::vector<int> host_labels(B);
    for (int i = 0; i < B; ++i) host_labels[i] = std::rand() % OUT;
    int* d_labels; cudaMalloc(&d_labels, B * sizeof(int));
    cudaMemcpy(d_labels, host_labels.data(), B * sizeof(int), cudaMemcpyHostToDevice);

    // Gradient buffers — one per parameter, allocated up front.
    Tensor* dW1 = mktensor({HID, IN});
    Tensor* db1 = mktensor({HID});
    Tensor* dW2 = mktensor({OUT, HID});
    Tensor* db2 = mktensor({OUT});
    Tensor* d_ln_g = mktensor({HID});
    Tensor* d_ln_b = mktensor({HID});

    Tensor* d_logits = mktensor({B, OUT});
    Tensor* d_h1_a   = mktensor({B, HID});
    Tensor* d_h1_ln  = mktensor({B, HID});
    Tensor* d_h1     = mktensor({B, HID});
    Tensor* d_x      = mktensor({B, IN});

    // ------ forward ---------------------------------------------------
    linear_forward(ctx, x, W1, b1, h1);
    layernorm_forward(h1, ln_g, ln_b, ln_mean, ln_istd, h1_ln, 1e-5f);
    gelu_forward(h1_ln, h1_a);
    linear_forward(ctx, h1_a, W2, b2, logits);

    float loss = cross_entropy_loss(logits, d_labels, B, OUT);
    std::printf("loss = %f\n", loss);

    // ------ backward --------------------------------------------------
    cross_entropy_backward(logits, d_labels, d_logits, B, OUT);
    linear_backward(ctx, d_logits, h1_a, W2, d_h1_a, dW2, db2);
    gelu_backward(d_h1_a, h1_ln, d_h1_ln);
    layernorm_backward(d_h1_ln, h1, ln_g, ln_mean, ln_istd,
                       d_h1, d_ln_g, d_ln_b);
    linear_backward(ctx, d_h1, x, W1, d_x, dW1, db1);

    std::printf("backward pass completed.\n");

    // ------ cleanup ---------------------------------------------------
    for (Tensor* t : {x, h1, h1_ln, h1_a, logits,
                      W1, b1, W2, b2, ln_g, ln_b,
                      ln_mean, ln_istd,
                      dW1, db1, dW2, db2, d_ln_g, d_ln_b,
                      d_logits, d_h1_a, d_h1_ln, d_h1, d_x}) {
        free_tensor(t);
    }
    cudaFree(d_labels);
    destroy_context(ctx);
    return 0;
}
