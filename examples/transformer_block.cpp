// A minimal GPT-style transformer block trained for a few steps on random
// tokens. The point: embedding, causal multi-head attention, residual adds,
// norms, and the MLP are composed purely from recorded ops, so the entire
// backward pass — through two matmuls, softmax, permutes, and four linears
// per attention call — is derived by autograd with no hand-written gradient
// code anywhere in this file.

#include "../ultraml.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace ultraml;

int main() {
    const int B = 4, T = 8, E = 32, H = 4, VOCAB = 50, STEPS = 30;

    NNContext* ctx = create_context();
    nn::manual_seed(0);

    nn::Embedding          embed(VOCAB, E);
    nn::MultiHeadAttention attn(E, H, /*causal=*/true, /*dropout_p=*/0.0f);
    nn::LayerNorm          ln1(E), ln2(E);
    nn::Sequential         mlp({
        new nn::Linear(E, 4 * E),
        new nn::GELU(),
        new nn::Linear(4 * E, E),
    });
    nn::Linear             head(E, VOCAB);

    std::vector<Tensor*> params;
    embed.collect_parameters(params);
    attn.collect_parameters(params);
    ln1.collect_parameters(params);
    ln2.collect_parameters(params);
    mlp.collect_parameters(params);
    head.collect_parameters(params);
    optim::Adam opt(params, 1e-3f);

    // Fixed random token stream; targets are the next token. Training just
    // memorizes it — loss falling from ~ln(VOCAB) shows gradients flow.
    std::srand(123);
    std::vector<int> h_ids(B * T), h_tgt(B * T);
    for (int i = 0; i < B * T; ++i) {
        h_ids[i] = std::rand() % VOCAB;
        h_tgt[i] = std::rand() % VOCAB;
    }
    int *d_ids, *d_tgt;
    ULTRAML_CUDA_CHECK(cudaMalloc(&d_ids, B * T * sizeof(int)));
    ULTRAML_CUDA_CHECK(cudaMalloc(&d_tgt, B * T * sizeof(int)));
    ULTRAML_CUDA_CHECK(cudaMemcpy(d_ids, h_ids.data(), B * T * sizeof(int),
                                  cudaMemcpyHostToDevice));
    ULTRAML_CUDA_CHECK(cudaMemcpy(d_tgt, h_tgt.data(), B * T * sizeof(int),
                                  cudaMemcpyHostToDevice));

    for (int step = 1; step <= STEPS; ++step) {
        Tensor* tok = embed.forward_ids(ctx, d_ids, B * T);   // [B*T, E]
        Tensor* x   = ops::reshape(tok, {B, T, E});

        // pre-norm block: x = x + Attn(LN(x)); x = x + MLP(LN(x))
        Tensor* a  = attn.forward(ctx, ln1.forward(ctx, x));
        x          = ops::add(x, a);
        Tensor* xf = ops::reshape(x, {B * T, E});
        Tensor* m  = mlp.forward(ctx, ln2.forward(ctx, xf));
        xf         = ops::add(xf, m);

        Tensor* logits = head.forward(ctx, xf);               // [B*T, VOCAB]
        Tensor* loss   = ops::cross_entropy(logits, d_tgt, B * T, VOCAB);

        if (step == 1 || step % 5 == 0)
            std::printf("step %2d  loss %.4f\n", step, ops::item(loss));

        autograd::backward(ctx, loss);
        opt.step();
        opt.zero_grad();
        autograd::clear();
    }

    ULTRAML_CUDA_CHECK(cudaFree(d_ids));
    ULTRAML_CUDA_CHECK(cudaFree(d_tgt));
    destroy_context(ctx);
    return 0;
}
