#include "ops.h"
#include "autograd.h"
#include "../core/macros.h"

#include "../activations/activations.h"
#include "../norms/norms.h"
#include "../losses/losses.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ultraml {
namespace ops {

namespace {

Tensor* fresh(const int* shape, int ndim) {
    return autograd::own(create_tensor(shape, ndim, false));
}

Tensor* like(const Tensor* x) { return fresh(x->shape, x->ndim); }

float read_scalar(const Tensor* t) {
    float v;
    ULTRAML_CUDA_CHECK(cudaMemcpy(&v, t->data, sizeof(float),
                                  cudaMemcpyDeviceToHost));
    return v;
}

// Fresh dropout seed per mask; any fixed stream of distinct values works.
unsigned long long next_seed() {
    static unsigned long long counter = 0x5DEECE66DULL;
    counter += 0x9E3779B97F4A7C15ULL;
    return counter;
}

// Reads [n, c, h, w] out of a cuDNN 4D tensor descriptor.
void desc_dims(cudnnTensorDescriptor_t desc, int shape[4]) {
    cudnnDataType_t dt;
    int ns, cs, hs, ws;
    ULTRAML_CUDNN_CHECK(cudnnGetTensor4dDescriptor(
        desc, &dt, &shape[0], &shape[1], &shape[2], &shape[3],
        &ns, &cs, &hs, &ws));
}

} // namespace

// ==================== activations ====================================
// The saved tensor matches what each kernel's backward wants: some take the
// forward input, others the forward output (see activations.h).
#define OPS_EW_SAVED_INPUT(opname, fwd, bwd)                                   \
    Tensor* opname(Tensor* x) {                                               \
        Tensor* y = like(x);                                                  \
        fwd(x, y);                                                            \
        autograd::record(#opname, {x}, y,                                     \
            [x](NNContext*, const Tensor* gy, std::vector<Tensor*>& gi) {     \
                if (gi[0]) bwd(gy, x, gi[0]);                                 \
            });                                                               \
        return y;                                                             \
    }

#define OPS_EW_SAVED_OUTPUT(opname, fwd, bwd)                                  \
    Tensor* opname(Tensor* x) {                                               \
        Tensor* y = like(x);                                                  \
        fwd(x, y);                                                            \
        autograd::record(#opname, {x}, y,                                     \
            [y](NNContext*, const Tensor* gy, std::vector<Tensor*>& gi) {     \
                if (gi[0]) bwd(gy, y, gi[0]);                                 \
            });                                                               \
        return y;                                                             \
    }

OPS_EW_SAVED_INPUT (relu,        relu_forward,        relu_backward)
OPS_EW_SAVED_OUTPUT(sigmoid,     sigmoid_forward,     sigmoid_backward)
OPS_EW_SAVED_OUTPUT(tanh,        tanh_forward,        tanh_backward)
OPS_EW_SAVED_INPUT (gelu,        gelu_forward,        gelu_backward)
OPS_EW_SAVED_INPUT (silu,        silu_forward,        silu_backward)
OPS_EW_SAVED_INPUT (softplus,    softplus_forward,    softplus_backward)
OPS_EW_SAVED_INPUT (mish,        mish_forward,        mish_backward)
OPS_EW_SAVED_INPUT (hardsigmoid, hardsigmoid_forward, hardsigmoid_backward)
OPS_EW_SAVED_INPUT (hardswish,   hardswish_forward,   hardswish_backward)
OPS_EW_SAVED_OUTPUT(softmax,     softmax_forward,     softmax_backward)
OPS_EW_SAVED_OUTPUT(log_softmax, log_softmax_forward, log_softmax_backward)

#undef OPS_EW_SAVED_INPUT
#undef OPS_EW_SAVED_OUTPUT

Tensor* leaky_relu(Tensor* x, float alpha) {
    Tensor* y = like(x);
    leaky_relu_forward(x, y, alpha);
    autograd::record("leaky_relu", {x}, y,
        [x, alpha](NNContext*, const Tensor* gy, std::vector<Tensor*>& gi) {
            if (gi[0]) leaky_relu_backward(gy, x, gi[0], alpha);
        });
    return y;
}

Tensor* elu(Tensor* x, float alpha) {
    Tensor* y = like(x);
    elu_forward(x, y, alpha);
    autograd::record("elu", {x}, y,
        [x, alpha](NNContext*, const Tensor* gy, std::vector<Tensor*>& gi) {
            if (gi[0]) elu_backward(gy, x, gi[0], alpha);
        });
    return y;
}

Tensor* hardtanh(Tensor* x, float min_val, float max_val) {
    Tensor* y = like(x);
    hardtanh_forward(x, y, min_val, max_val);
    autograd::record("hardtanh", {x}, y,
        [x, min_val, max_val](NNContext*, const Tensor* gy,
                              std::vector<Tensor*>& gi) {
            if (gi[0]) hardtanh_backward(gy, x, gi[0], min_val, max_val);
        });
    return y;
}

// ==================== layers =========================================

Tensor* linear(NNContext* ctx, Tensor* x, Tensor* W, Tensor* b) {
    int shape[2] = { x->shape[0], W->shape[0] };
    Tensor* y = fresh(shape, 2);
    linear_forward(ctx, x, W, b, y);
    autograd::record("linear", {x, W, b}, y,
        [x, W](NNContext* c, const Tensor* gy, std::vector<Tensor*>& gi) {
            if (gi[0] || gi[1] || gi[2])
                linear_backward(c, gy, x, W, gi[0], gi[1], gi[2]);
        });
    return y;
}

Tensor* conv2d(NNContext* ctx, ConvDescriptor* desc,
               Tensor* x, Tensor* W, Tensor* b) {
    int shape[4];
    desc_dims(desc->output_desc, shape);
    Tensor* y = fresh(shape, 4);
    conv2d_forward(ctx, desc, x, W, b, y);
    autograd::record("conv2d", {x, W, b}, y,
        [desc, x, W](NNContext* c, const Tensor* gy, std::vector<Tensor*>& gi) {
            if (gi[0] || gi[1] || gi[2])
                conv2d_backward(c, desc, gy, x, W, gi[0], gi[1], gi[2]);
        });
    return y;
}

Tensor* pool2d(NNContext* ctx, PoolDescriptor* desc, Tensor* x) {
    int shape[4];
    desc_dims(desc->output_desc, shape);
    Tensor* y = fresh(shape, 4);
    pool2d_forward(ctx, desc, x, y);
    autograd::record("pool2d", {x}, y,
        [desc, x, y](NNContext* c, const Tensor* gy, std::vector<Tensor*>& gi) {
            if (gi[0]) pool2d_backward(c, desc, gy, y, x, gi[0]);
        });
    return y;
}

// ==================== norms ==========================================
// The norm backward kernels always write grad_input, so hand them a
// throwaway buffer when the input itself does not need gradients.

Tensor* layernorm(Tensor* x, Tensor* gamma, Tensor* beta, float eps) {
    int cols = x->shape[x->ndim - 1];
    int rows = x->size / cols;
    Tensor* mean = fresh(&rows, 1);
    Tensor* istd = fresh(&rows, 1);
    Tensor* y = like(x);
    layernorm_forward(x, gamma, beta, mean, istd, y, eps);
    autograd::record("layernorm", {x, gamma, beta}, y,
        [x, gamma, mean, istd](NNContext*, const Tensor* gy,
                               std::vector<Tensor*>& gi) {
            Tensor* gx = gi[0] ? gi[0] : create_tensor(x->shape, x->ndim, false);
            layernorm_backward(gy, x, gamma, mean, istd, gx, gi[1], gi[2]);
            if (!gi[0]) free_tensor(gx);
        });
    return y;
}

Tensor* rmsnorm(Tensor* x, Tensor* gamma, float eps) {
    int cols = x->shape[x->ndim - 1];
    int rows = x->size / cols;
    Tensor* rrms = fresh(&rows, 1);
    Tensor* y = like(x);
    rmsnorm_forward(x, gamma, rrms, y, eps);
    autograd::record("rmsnorm", {x, gamma}, y,
        [x, gamma, rrms](NNContext*, const Tensor* gy,
                         std::vector<Tensor*>& gi) {
            Tensor* gx = gi[0] ? gi[0] : create_tensor(x->shape, x->ndim, false);
            rmsnorm_backward(gy, x, gamma, rrms, gx, gi[1]);
            if (!gi[0]) free_tensor(gx);
        });
    return y;
}

Tensor* groupnorm(Tensor* x, Tensor* gamma, Tensor* beta, int num_groups,
                  float eps) {
    int sshape[2] = { x->shape[0], num_groups };
    Tensor* mean = fresh(sshape, 2);
    Tensor* istd = fresh(sshape, 2);
    Tensor* y = like(x);
    groupnorm_forward(x, gamma, beta, num_groups, mean, istd, y, eps);
    autograd::record("groupnorm", {x, gamma, beta}, y,
        [x, gamma, num_groups, mean, istd](NNContext*, const Tensor* gy,
                                           std::vector<Tensor*>& gi) {
            Tensor* gx = gi[0] ? gi[0] : create_tensor(x->shape, x->ndim, false);
            groupnorm_backward(gy, x, gamma, num_groups, mean, istd,
                               gx, gi[1], gi[2]);
            if (!gi[0]) free_tensor(gx);
        });
    return y;
}

namespace {

// Shared body for batchnorm1d/2d: stats are per feature/channel
// (= shape[1]). Eval mode computes without recording — the kernels save no
// statistics there, so no backward is possible.
template <typename FwdFn, typename BwdFn>
Tensor* batchnorm_impl(const char* name, FwdFn fwd, BwdFn bwd,
                       Tensor* x, Tensor* gamma, Tensor* beta,
                       Tensor* running_mean, Tensor* running_var,
                       float momentum, float eps, bool training) {
    Tensor* y = like(x);
    if (!training) {
        fwd(x, gamma, beta, running_mean, running_var,
            nullptr, nullptr, y, momentum, eps, false);
        return y;
    }
    int feats = x->shape[1];
    Tensor* mean = fresh(&feats, 1);
    Tensor* istd = fresh(&feats, 1);
    fwd(x, gamma, beta, running_mean, running_var,
        mean, istd, y, momentum, eps, true);
    autograd::record(name, {x, gamma, beta}, y,
        [bwd, x, gamma, mean, istd](NNContext*, const Tensor* gy,
                                    std::vector<Tensor*>& gi) {
            Tensor* gx = gi[0] ? gi[0] : create_tensor(x->shape, x->ndim, false);
            bwd(gy, x, gamma, mean, istd, gx, gi[1], gi[2]);
            if (!gi[0]) free_tensor(gx);
        });
    return y;
}

} // namespace

Tensor* batchnorm1d(Tensor* x, Tensor* gamma, Tensor* beta,
                    Tensor* running_mean, Tensor* running_var,
                    float momentum, float eps, bool training) {
    return batchnorm_impl("batchnorm1d",
                          batchnorm1d_forward, batchnorm1d_backward,
                          x, gamma, beta, running_mean, running_var,
                          momentum, eps, training);
}

Tensor* batchnorm2d(Tensor* x, Tensor* gamma, Tensor* beta,
                    Tensor* running_mean, Tensor* running_var,
                    float momentum, float eps, bool training) {
    return batchnorm_impl("batchnorm2d",
                          batchnorm2d_forward, batchnorm2d_backward,
                          x, gamma, beta, running_mean, running_var,
                          momentum, eps, training);
}

// ==================== element-wise / shape ===========================

Tensor* add(Tensor* a, Tensor* b) {
    Tensor* y = like(a);
    tensor_add(a, b, y);
    autograd::record("add", {a, b}, y,
        [](NNContext*, const Tensor* gy, std::vector<Tensor*>& gi) {
            if (gi[0]) copy_tensor(gi[0], gy);
            if (gi[1]) copy_tensor(gi[1], gy);
        });
    return y;
}

Tensor* mul(Tensor* a, Tensor* b) {
    Tensor* y = like(a);
    tensor_multiply(a, b, y);
    autograd::record("mul", {a, b}, y,
        [a, b](NNContext*, const Tensor* gy, std::vector<Tensor*>& gi) {
            if (gi[0]) tensor_multiply(gy, b, gi[0]);
            if (gi[1]) tensor_multiply(gy, a, gi[1]);
        });
    return y;
}

Tensor* scale(Tensor* x, float s) {
    Tensor* y = like(x);
    tensor_scale(x, s, y);
    autograd::record("scale", {x}, y,
        [s](NNContext*, const Tensor* gy, std::vector<Tensor*>& gi) {
            if (gi[0]) tensor_scale(gy, s, gi[0]);
        });
    return y;
}

Tensor* reshape(Tensor* x, const int* shape, int ndim) {
    Tensor* y = fresh(shape, ndim);
    if (y->size != x->size) {
        fprintf(stderr, "ops::reshape: size mismatch (%d -> %d)\n",
                x->size, y->size);
        exit(EXIT_FAILURE);
    }
    copy_tensor(y, x);
    autograd::record("reshape", {x}, y,
        [](NNContext*, const Tensor* gy, std::vector<Tensor*>& gi) {
            if (gi[0]) copy_tensor(gi[0], gy);
        });
    return y;
}

Tensor* reshape(Tensor* x, std::initializer_list<int> shape) {
    std::vector<int> s(shape);
    return reshape(x, s.data(), (int)s.size());
}

// ==================== new functionality ==============================

Tensor* dropout(Tensor* x, float p, bool training) {
    if (!training || p <= 0.0f) return x;
    Tensor* mask = like(x);
    dropout_make_mask(mask, p, next_seed());
    Tensor* y = like(x);
    tensor_multiply(x, mask, y);
    autograd::record("dropout", {x}, y,
        [mask](NNContext*, const Tensor* gy, std::vector<Tensor*>& gi) {
            if (gi[0]) tensor_multiply(gy, mask, gi[0]);
        });
    return y;
}

Tensor* embedding(Tensor* weight, const int* indices_device, int n_indices) {
    int shape[2] = { n_indices, weight->shape[weight->ndim - 1] };
    Tensor* y = fresh(shape, 2);
    embedding_forward(weight, indices_device, n_indices, y);
    autograd::record("embedding", {weight}, y,
        [indices_device, n_indices](NNContext*, const Tensor* gy,
                                    std::vector<Tensor*>& gi) {
            // gi[0] arrives zero-filled; the kernel scatter-adds into it.
            if (gi[0]) embedding_backward(gy, indices_device, n_indices, gi[0]);
        });
    return y;
}

Tensor* matmul(NNContext* ctx, Tensor* a, Tensor* b,
               bool trans_a, bool trans_b) {
    int a_rows = a->shape[a->ndim - 2], a_cols = a->shape[a->ndim - 1];
    int b_rows = b->shape[b->ndim - 2], b_cols = b->shape[b->ndim - 1];
    int M = trans_a ? a_cols : a_rows;
    int N = trans_b ? b_rows : b_cols;

    Tensor* y;
    if (a->ndim == 3) {
        int shape[3] = { a->shape[0], M, N };
        y = fresh(shape, 3);
    } else {
        int shape[2] = { M, N };
        y = fresh(shape, 2);
    }
    batched_matmul(ctx, a, b, y, trans_a, trans_b);
    autograd::record("matmul", {a, b}, y,
        [a, b, trans_a, trans_b](NNContext* c, const Tensor* gy,
                                 std::vector<Tensor*>& gi) {
            batched_matmul_backward(c, gy, a, b, trans_a, trans_b,
                                    gi[0], gi[1]);
        });
    return y;
}

Tensor* permute0213(Tensor* x) {
    int shape[4] = { x->shape[0], x->shape[2], x->shape[1], x->shape[3] };
    Tensor* y = fresh(shape, 4);
    permute_0213(x, y);
    autograd::record("permute0213", {x}, y,
        [](NNContext*, const Tensor* gy, std::vector<Tensor*>& gi) {
            // Swapping dims 1 and 2 twice is the identity, so the same
            // kernel maps the output-shaped gradient back to input shape.
            if (gi[0]) permute_0213(gy, gi[0]);
        });
    return y;
}

Tensor* causal_mask(Tensor* scores, int seq_len) {
    Tensor* y = like(scores);
    add_causal_mask(scores, y, seq_len);
    // The mask is an additive constant, so the gradient passes through.
    autograd::record("causal_mask", {scores}, y,
        [](NNContext*, const Tensor* gy, std::vector<Tensor*>& gi) {
            if (gi[0]) copy_tensor(gi[0], gy);
        });
    return y;
}

// ==================== losses =========================================
// Each returns a [1] tensor so losses compose (weighted sums, multi-task).
// The recorded backward scales by the incoming seed gradient, so a loss
// used inside a larger expression still differentiates correctly.

namespace {

Tensor* make_loss_scalar(float value) {
    int one = 1;
    Tensor* out = fresh(&one, 1);
    copy_from_host(out, &value);
    return out;
}

void scale_by_seed(const Tensor* gy, Tensor* grad) {
    float g = read_scalar(gy);
    if (g != 1.0f) tensor_scale(grad, g, grad);
}

} // namespace

Tensor* mse_loss(Tensor* pred, Tensor* target) {
    Tensor* out = make_loss_scalar(ultraml::mse_loss(pred, target));
    autograd::record("mse_loss", {pred}, out,
        [pred, target](NNContext*, const Tensor* gy,
                       std::vector<Tensor*>& gi) {
            if (!gi[0]) return;
            mse_loss_backward(pred, target, gi[0]);
            scale_by_seed(gy, gi[0]);
        });
    return out;
}

Tensor* l1_loss(Tensor* pred, Tensor* target) {
    Tensor* out = make_loss_scalar(ultraml::l1_loss(pred, target));
    autograd::record("l1_loss", {pred}, out,
        [pred, target](NNContext*, const Tensor* gy,
                       std::vector<Tensor*>& gi) {
            if (!gi[0]) return;
            l1_loss_backward(pred, target, gi[0]);
            scale_by_seed(gy, gi[0]);
        });
    return out;
}

Tensor* huber_loss(Tensor* pred, Tensor* target, float delta) {
    Tensor* out = make_loss_scalar(ultraml::huber_loss(pred, target, delta));
    autograd::record("huber_loss", {pred}, out,
        [pred, target, delta](NNContext*, const Tensor* gy,
                              std::vector<Tensor*>& gi) {
            if (!gi[0]) return;
            huber_loss_backward(pred, target, gi[0], delta);
            scale_by_seed(gy, gi[0]);
        });
    return out;
}

Tensor* cross_entropy(Tensor* logits, const int* targets_device,
                      int batch, int num_classes) {
    Tensor* out = make_loss_scalar(
        cross_entropy_loss(logits, targets_device, batch, num_classes));
    autograd::record("cross_entropy", {logits}, out,
        [logits, targets_device, batch, num_classes](
            NNContext*, const Tensor* gy, std::vector<Tensor*>& gi) {
            if (!gi[0]) return;
            cross_entropy_backward(logits, targets_device, gi[0],
                                   batch, num_classes);
            scale_by_seed(gy, gi[0]);
        });
    return out;
}

Tensor* bce_with_logits(Tensor* logits, Tensor* target) {
    Tensor* out = make_loss_scalar(bce_with_logits_loss(logits, target));
    autograd::record("bce_with_logits", {logits}, out,
        [logits, target](NNContext*, const Tensor* gy,
                         std::vector<Tensor*>& gi) {
            if (!gi[0]) return;
            bce_with_logits_backward(logits, target, gi[0]);
            scale_by_seed(gy, gi[0]);
        });
    return out;
}

float item(const Tensor* scalar) {
    return read_scalar(scalar);
}

} // namespace ops
} // namespace ultraml
