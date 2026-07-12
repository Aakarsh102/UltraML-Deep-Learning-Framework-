#include "nn.h"
#include "init.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace ultraml {
namespace nn {

namespace {

Tensor* param(std::initializer_list<int> shape) {
    std::vector<int> s(shape);
    return create_tensor(s.data(), (int)s.size(), /*requires_grad=*/true);
}

Tensor* buffer(std::initializer_list<int> shape) {
    std::vector<int> s(shape);
    return create_tensor(s.data(), (int)s.size(), /*requires_grad=*/false);
}

void require_ndim(const Tensor* x, int ndim, const char* who) {
    if (x->ndim != ndim) {
        fprintf(stderr, "%s: expected %dD input, got ndim=%d\n",
                who, ndim, x->ndim);
        exit(EXIT_FAILURE);
    }
}

} // namespace

// ==================== Sequential =====================================

Sequential::Sequential(std::initializer_list<Module*> modules)
    : children(modules) {}

Sequential::~Sequential() {
    for (Module* m : children) delete m;
}

Sequential* Sequential::add(Module* module) {
    children.push_back(module);
    return this;
}

Tensor* Sequential::forward(NNContext* ctx, Tensor* x) {
    for (Module* m : children) x = m->forward(ctx, x);
    return x;
}

void Sequential::collect_parameters(std::vector<Tensor*>& out) {
    for (Module* m : children) m->collect_parameters(out);
}

void Sequential::set_training(bool training) {
    Module::set_training(training);
    for (Module* m : children) m->set_training(training);
}

// ==================== Linear =========================================

Linear::Linear(int in_features, int out_features, bool with_bias)
    : in_features(in_features), out_features(out_features) {
    weight = param({out_features, in_features});
    float bound = 1.0f / std::sqrt((float)in_features);
    init_uniform(weight, -bound, bound);
    if (with_bias) {
        bias = param({out_features});
        init_uniform(bias, -bound, bound);
    } else {
        bias = nullptr;
    }
}

Linear::~Linear() {
    free_tensor(weight);
    if (bias) free_tensor(bias);
}

Tensor* Linear::forward(NNContext* ctx, Tensor* x) {
    require_ndim(x, 2, "nn::Linear");
    return ops::linear(ctx, x, weight, bias);
}

void Linear::collect_parameters(std::vector<Tensor*>& out) {
    out.push_back(weight);
    if (bias) out.push_back(bias);
}

// ==================== Conv2d =========================================

Conv2d::Conv2d(int in_channels, int out_channels,
               int kernel_h, int kernel_w,
               int stride_h, int stride_w,
               int pad_h, int pad_w,
               bool with_bias)
    : in_channels(in_channels), out_channels(out_channels),
      kernel_h(kernel_h), kernel_w(kernel_w),
      stride_h(stride_h), stride_w(stride_w),
      pad_h(pad_h), pad_w(pad_w) {
    weight = param({out_channels, in_channels, kernel_h, kernel_w});
    float bound = 1.0f / std::sqrt((float)(in_channels * kernel_h * kernel_w));
    init_uniform(weight, -bound, bound);
    if (with_bias) {
        bias = param({out_channels});
        init_uniform(bias, -bound, bound);
    } else {
        bias = nullptr;
    }
}

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size,
               int stride, int padding, bool with_bias)
    : Conv2d(in_channels, out_channels, kernel_size, kernel_size,
             stride, stride, padding, padding, with_bias) {}

Conv2d::~Conv2d() {
    free_tensor(weight);
    if (bias) free_tensor(bias);
    if (desc_) free_conv_descriptor(desc_);
}

Tensor* Conv2d::forward(NNContext* ctx, Tensor* x) {
    require_ndim(x, 4, "nn::Conv2d");
    if (x->shape[1] != in_channels) {
        fprintf(stderr, "nn::Conv2d: expected %d input channels, got %d\n",
                in_channels, x->shape[1]);
        exit(EXIT_FAILURE);
    }
    int n = x->shape[0], h = x->shape[2], w = x->shape[3];
    if (!desc_ || n != cached_n_ || h != cached_h_ || w != cached_w_) {
        if (desc_) free_conv_descriptor(desc_);
        desc_ = create_conv_descriptor(ctx, n, in_channels, h, w,
                                       out_channels, kernel_h, kernel_w,
                                       stride_h, stride_w, pad_h, pad_w);
        cached_n_ = n; cached_h_ = h; cached_w_ = w;
    }
    return ops::conv2d(ctx, desc_, x, weight, bias);
}

void Conv2d::collect_parameters(std::vector<Tensor*>& out) {
    out.push_back(weight);
    if (bias) out.push_back(bias);
}

// ==================== Pooling ========================================

Pool2d::Pool2d(int window, int stride, int padding, cudnnPoolingMode_t mode)
    : window(window), stride(stride), padding(padding), mode_(mode) {}

Pool2d::~Pool2d() {
    if (desc_) free_pool_descriptor(desc_);
}

Tensor* Pool2d::forward(NNContext* ctx, Tensor* x) {
    require_ndim(x, 4, "nn::Pool2d");
    int n = x->shape[0], c = x->shape[1], h = x->shape[2], w = x->shape[3];
    if (!desc_ || n != cached_n_ || c != cached_c_ ||
        h != cached_h_ || w != cached_w_) {
        if (desc_) free_pool_descriptor(desc_);
        desc_ = create_pool_descriptor(n, c, h, w,
                                       window, window, stride, stride,
                                       padding, padding, mode_);
        cached_n_ = n; cached_c_ = c; cached_h_ = h; cached_w_ = w;
    }
    return ops::pool2d(ctx, desc_, x);
}

MaxPool2d::MaxPool2d(int window, int stride, int padding)
    : Pool2d(window, stride > 0 ? stride : window, padding,
             CUDNN_POOLING_MAX) {}

AvgPool2d::AvgPool2d(int window, int stride, int padding)
    : Pool2d(window, stride > 0 ? stride : window, padding,
             CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING) {}

// ==================== norms ==========================================

LayerNorm::LayerNorm(int dim, float eps) : eps(eps) {
    gamma = param({dim}); init_ones(gamma);
    beta  = param({dim}); init_zeros(beta);
}
LayerNorm::~LayerNorm() {
    free_tensor(gamma);
    free_tensor(beta);
}
Tensor* LayerNorm::forward(NNContext*, Tensor* x) {
    return ops::layernorm(x, gamma, beta, eps);
}
void LayerNorm::collect_parameters(std::vector<Tensor*>& out) {
    out.push_back(gamma);
    out.push_back(beta);
}

RMSNorm::RMSNorm(int dim, float eps) : eps(eps) {
    gamma = param({dim}); init_ones(gamma);
}
RMSNorm::~RMSNorm() { free_tensor(gamma); }
Tensor* RMSNorm::forward(NNContext*, Tensor* x) {
    return ops::rmsnorm(x, gamma, eps);
}
void RMSNorm::collect_parameters(std::vector<Tensor*>& out) {
    out.push_back(gamma);
}

GroupNorm::GroupNorm(int num_groups, int channels, float eps)
    : num_groups(num_groups), eps(eps) {
    gamma = param({channels}); init_ones(gamma);
    beta  = param({channels}); init_zeros(beta);
}
GroupNorm::~GroupNorm() {
    free_tensor(gamma);
    free_tensor(beta);
}
Tensor* GroupNorm::forward(NNContext*, Tensor* x) {
    return ops::groupnorm(x, gamma, beta, num_groups, eps);
}
void GroupNorm::collect_parameters(std::vector<Tensor*>& out) {
    out.push_back(gamma);
    out.push_back(beta);
}

BatchNorm::BatchNorm(int features, int dims, float momentum, float eps)
    : momentum(momentum), eps(eps), dims_(dims) {
    gamma        = param({features});  init_ones(gamma);
    beta         = param({features});  init_zeros(beta);
    running_mean = buffer({features}); init_zeros(running_mean);
    running_var  = buffer({features}); init_ones(running_var);
}
BatchNorm::~BatchNorm() {
    free_tensor(gamma);
    free_tensor(beta);
    free_tensor(running_mean);
    free_tensor(running_var);
}
Tensor* BatchNorm::forward(NNContext*, Tensor* x) {
    if (dims_ == 1) {
        return ops::batchnorm1d(x, gamma, beta, running_mean, running_var,
                                momentum, eps, is_training());
    }
    return ops::batchnorm2d(x, gamma, beta, running_mean, running_var,
                            momentum, eps, is_training());
}
void BatchNorm::collect_parameters(std::vector<Tensor*>& out) {
    out.push_back(gamma);
    out.push_back(beta);
}

BatchNorm1d::BatchNorm1d(int features, float momentum, float eps)
    : BatchNorm(features, 1, momentum, eps) {}
BatchNorm2d::BatchNorm2d(int channels, float momentum, float eps)
    : BatchNorm(channels, 2, momentum, eps) {}

// ==================== Embedding ======================================

Embedding::Embedding(int num_embeddings, int embedding_dim)
    : num_embeddings(num_embeddings), embedding_dim(embedding_dim) {
    weight = param({num_embeddings, embedding_dim});
    init_normal(weight, 0.0f, 1.0f);
}
Embedding::~Embedding() { free_tensor(weight); }

Tensor* Embedding::forward_ids(NNContext*, const int* ids_device, int n) {
    return ops::embedding(weight, ids_device, n);
}

Tensor* Embedding::forward(NNContext*, Tensor*) {
    fprintf(stderr,
            "nn::Embedding: use forward_ids(ctx, ids_device, n) — "
            "embeddings consume integer ids, not a float tensor\n");
    exit(EXIT_FAILURE);
}

void Embedding::collect_parameters(std::vector<Tensor*>& out) {
    out.push_back(weight);
}

// ==================== MultiHeadAttention =============================

MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads,
                                       bool causal, float dropout_p)
    : embed_dim(embed_dim), num_heads(num_heads),
      head_dim(embed_dim / num_heads), causal(causal), dropout_p(dropout_p) {
    if (embed_dim % num_heads != 0) {
        fprintf(stderr,
                "nn::MultiHeadAttention: embed_dim (%d) must be divisible "
                "by num_heads (%d)\n", embed_dim, num_heads);
        exit(EXIT_FAILURE);
    }
    q_proj   = new Linear(embed_dim, embed_dim);
    k_proj   = new Linear(embed_dim, embed_dim);
    v_proj   = new Linear(embed_dim, embed_dim);
    out_proj = new Linear(embed_dim, embed_dim);
}

MultiHeadAttention::~MultiHeadAttention() {
    delete q_proj;
    delete k_proj;
    delete v_proj;
    delete out_proj;
}

Tensor* MultiHeadAttention::forward(NNContext* ctx, Tensor* x) {
    require_ndim(x, 3, "nn::MultiHeadAttention");
    int B = x->shape[0], T = x->shape[1], E = x->shape[2];
    int H = num_heads, Dh = head_dim;

    Tensor* x2 = ops::reshape(x, {B * T, E});
    Tensor* q  = q_proj->forward(ctx, x2);    // [B*T, E]
    Tensor* k  = k_proj->forward(ctx, x2);
    Tensor* v  = v_proj->forward(ctx, x2);

    // [B*T, E] -> [B, T, H, Dh] -> [B, H, T, Dh] -> [B*H, T, Dh]
    auto split_heads = [&](Tensor* t) {
        Tensor* t4 = ops::reshape(t, {B, T, H, Dh});
        Tensor* tp = ops::permute0213(t4);
        return ops::reshape(tp, {B * H, T, Dh});
    };
    Tensor* qh = split_heads(q);
    Tensor* kh = split_heads(k);
    Tensor* vh = split_heads(v);

    // scaled dot-product attention over each head
    Tensor* scores = ops::matmul(ctx, qh, kh, false, true);     // [B*H, T, T]
    scores = ops::scale(scores, 1.0f / std::sqrt((float)Dh));

    Tensor* rows = ops::reshape(scores, {B * H * T, T});
    if (causal) rows = ops::causal_mask(rows, T);
    Tensor* attn = ops::softmax(rows);
    attn = ops::dropout(attn, dropout_p, is_training());
    Tensor* attn3 = ops::reshape(attn, {B * H, T, T});

    Tensor* heads = ops::matmul(ctx, attn3, vh, false, false);  // [B*H, T, Dh]

    // merge heads: [B*H, T, Dh] -> [B, H, T, Dh] -> [B, T, H, Dh] -> [B*T, E]
    Tensor* h4 = ops::reshape(heads, {B, H, T, Dh});
    Tensor* hp = ops::permute0213(h4);
    Tensor* merged = ops::reshape(hp, {B * T, E});

    Tensor* out = out_proj->forward(ctx, merged);               // [B*T, E]
    return ops::reshape(out, {B, T, E});
}

void MultiHeadAttention::collect_parameters(std::vector<Tensor*>& out) {
    q_proj->collect_parameters(out);
    k_proj->collect_parameters(out);
    v_proj->collect_parameters(out);
    out_proj->collect_parameters(out);
}

} // namespace nn
} // namespace ultraml
