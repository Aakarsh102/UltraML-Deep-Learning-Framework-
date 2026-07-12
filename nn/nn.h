#pragma once

// Modular layer system on top of the autograd ops.
//
// A Module owns its parameters (Tensors created with requires_grad = true)
// and composes ops:: calls in forward(). Because every ops:: call records
// itself on the autograd tape, Module::forward needs no matching backward —
// autograd::backward() derives it. To add a new layer: subclass Module,
// allocate parameters in the constructor, chain ops:: (or your own recorded
// ops) in forward(), and report parameters in collect_parameters(). It then
// works inside Sequential and with every optimizer automatically.
//
// train()/eval() toggles training behaviour (Dropout, BatchNorm running
// stats) for a module and all its children.

#include "../core/tensor.h"
#include "../core/context.h"
#include "../layers/layers.h"
#include "../autograd/ops.h"

#include <initializer_list>
#include <vector>

namespace ultraml {
namespace nn {

// ==================== base ===========================================
struct Module {
    virtual ~Module() = default;

    virtual Tensor* forward(NNContext* ctx, Tensor* x) = 0;

    // Append this module's (and children's) parameters to `out`.
    virtual void collect_parameters(std::vector<Tensor*>& out) { (void)out; }

    virtual void set_training(bool training) { training_ = training; }

    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> out;
        collect_parameters(out);
        return out;
    }
    void train() { set_training(true); }
    void eval()  { set_training(false); }
    bool is_training() const { return training_; }

  protected:
    bool training_ = true;
};

// ==================== containers =====================================
// Owns its children and frees them on destruction.
struct Sequential : Module {
    Sequential() = default;
    Sequential(std::initializer_list<Module*> modules);
    ~Sequential() override;

    Sequential* add(Module* module);   // takes ownership; returns this

    Tensor* forward(NNContext* ctx, Tensor* x) override;
    void collect_parameters(std::vector<Tensor*>& out) override;
    void set_training(bool training) override;

    std::vector<Module*> children;
};

// ==================== core layers ====================================
struct Linear : Module {
    Linear(int in_features, int out_features, bool with_bias = true);
    ~Linear() override;

    Tensor* forward(NNContext* ctx, Tensor* x) override;   // x [batch, in]
    void collect_parameters(std::vector<Tensor*>& out) override;

    Tensor* weight;   // [out, in]
    Tensor* bias;     // [out] or nullptr
    int in_features, out_features;
};

// Input [N, C, H, W]. The cuDNN descriptor is (re)built lazily whenever the
// input geometry changes, so one module handles varying batch sizes.
struct Conv2d : Module {
    Conv2d(int in_channels, int out_channels,
           int kernel_h, int kernel_w,
           int stride_h = 1, int stride_w = 1,
           int pad_h = 0, int pad_w = 0,
           bool with_bias = true);
    Conv2d(int in_channels, int out_channels, int kernel_size,
           int stride = 1, int padding = 0, bool with_bias = true);
    ~Conv2d() override;

    Tensor* forward(NNContext* ctx, Tensor* x) override;
    void collect_parameters(std::vector<Tensor*>& out) override;

    Tensor* weight;   // [out_c, in_c, kh, kw]
    Tensor* bias;     // [out_c] or nullptr
    int in_channels, out_channels;
    int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w;

  private:
    ConvDescriptor* desc_ = nullptr;
    int cached_n_ = -1, cached_h_ = -1, cached_w_ = -1;
};

// Shared pooling machinery; descriptor cached per input geometry.
struct Pool2d : Module {
    Pool2d(int window, int stride, int padding, cudnnPoolingMode_t mode);
    ~Pool2d() override;
    Tensor* forward(NNContext* ctx, Tensor* x) override;

    int window, stride, padding;

  private:
    cudnnPoolingMode_t mode_;
    PoolDescriptor* desc_ = nullptr;
    int cached_n_ = -1, cached_c_ = -1, cached_h_ = -1, cached_w_ = -1;
};

struct MaxPool2d : Pool2d {
    explicit MaxPool2d(int window, int stride = -1, int padding = 0);
};
struct AvgPool2d : Pool2d {
    explicit AvgPool2d(int window, int stride = -1, int padding = 0);
};

// ==================== activations ====================================
#define ULTRAML_NN_ACTIVATION(Name, opfn)                                      \
    struct Name : Module {                                                     \
        Tensor* forward(NNContext*, Tensor* x) override {                      \
            return ops::opfn(x);                                               \
        }                                                                      \
    };

ULTRAML_NN_ACTIVATION(ReLU,        relu)
ULTRAML_NN_ACTIVATION(Sigmoid,     sigmoid)
ULTRAML_NN_ACTIVATION(Tanh,        tanh)
ULTRAML_NN_ACTIVATION(GELU,        gelu)
ULTRAML_NN_ACTIVATION(SiLU,        silu)
ULTRAML_NN_ACTIVATION(Softplus,    softplus)
ULTRAML_NN_ACTIVATION(Mish,        mish)
ULTRAML_NN_ACTIVATION(HardSigmoid, hardsigmoid)
ULTRAML_NN_ACTIVATION(HardSwish,   hardswish)
ULTRAML_NN_ACTIVATION(Softmax,     softmax)
ULTRAML_NN_ACTIVATION(LogSoftmax,  log_softmax)

#undef ULTRAML_NN_ACTIVATION

struct LeakyReLU : Module {
    explicit LeakyReLU(float alpha = 0.01f) : alpha(alpha) {}
    Tensor* forward(NNContext*, Tensor* x) override {
        return ops::leaky_relu(x, alpha);
    }
    float alpha;
};

struct ELU : Module {
    explicit ELU(float alpha = 1.0f) : alpha(alpha) {}
    Tensor* forward(NNContext*, Tensor* x) override {
        return ops::elu(x, alpha);
    }
    float alpha;
};

struct HardTanh : Module {
    HardTanh(float min_val = -1.0f, float max_val = 1.0f)
        : min_val(min_val), max_val(max_val) {}
    Tensor* forward(NNContext*, Tensor* x) override {
        return ops::hardtanh(x, min_val, max_val);
    }
    float min_val, max_val;
};

// ==================== norms ==========================================
struct LayerNorm : Module {
    explicit LayerNorm(int dim, float eps = 1e-5f);
    ~LayerNorm() override;
    Tensor* forward(NNContext* ctx, Tensor* x) override;
    void collect_parameters(std::vector<Tensor*>& out) override;

    Tensor* gamma;    // [dim]
    Tensor* beta;     // [dim]
    float eps;
};

struct RMSNorm : Module {
    explicit RMSNorm(int dim, float eps = 1e-5f);
    ~RMSNorm() override;
    Tensor* forward(NNContext* ctx, Tensor* x) override;
    void collect_parameters(std::vector<Tensor*>& out) override;

    Tensor* gamma;    // [dim]
    float eps;
};

struct GroupNorm : Module {
    GroupNorm(int num_groups, int channels, float eps = 1e-5f);
    ~GroupNorm() override;
    Tensor* forward(NNContext* ctx, Tensor* x) override;
    void collect_parameters(std::vector<Tensor*>& out) override;

    Tensor* gamma;    // [channels]
    Tensor* beta;     // [channels]
    int num_groups;
    float eps;
};

// Shared batchnorm state; `dims` picks the 1d/2d kernel.
struct BatchNorm : Module {
    BatchNorm(int features, int dims, float momentum, float eps);
    ~BatchNorm() override;
    Tensor* forward(NNContext* ctx, Tensor* x) override;
    void collect_parameters(std::vector<Tensor*>& out) override;

    Tensor* gamma;          // [features], learnable
    Tensor* beta;           // [features], learnable
    Tensor* running_mean;   // [features], buffer
    Tensor* running_var;    // [features], buffer
    float momentum, eps;

  private:
    int dims_;
};

struct BatchNorm1d : BatchNorm {
    explicit BatchNorm1d(int features, float momentum = 0.1f,
                         float eps = 1e-5f);
};
struct BatchNorm2d : BatchNorm {
    explicit BatchNorm2d(int channels, float momentum = 0.1f,
                         float eps = 1e-5f);
};

// ==================== regularisation / embedding / shape ============
struct Dropout : Module {
    explicit Dropout(float p = 0.5f) : p(p) {}
    Tensor* forward(NNContext*, Tensor* x) override {
        return ops::dropout(x, p, is_training());
    }
    float p;
};

struct Embedding : Module {
    Embedding(int num_embeddings, int embedding_dim);
    ~Embedding() override;

    // ids_device: n int32 token ids on the device -> [n, embedding_dim].
    Tensor* forward_ids(NNContext* ctx, const int* ids_device, int n);

    // Embedding consumes integer ids, not a float tensor.
    Tensor* forward(NNContext* ctx, Tensor* x) override;
    void collect_parameters(std::vector<Tensor*>& out) override;

    Tensor* weight;   // [num_embeddings, embedding_dim]
    int num_embeddings, embedding_dim;
};

// [N, ...] -> [N, prod(...)]
struct Flatten : Module {
    Tensor* forward(NNContext*, Tensor* x) override {
        return ops::reshape(x, { x->shape[0], x->size / x->shape[0] });
    }
};

// ==================== attention ======================================
// Standard multi-head self-attention, built entirely from recorded ops
// (linear, matmul, permute, softmax, dropout) — its backward pass is fully
// derived by autograd, which is the pattern for composing new blocks.
struct MultiHeadAttention : Module {
    MultiHeadAttention(int embed_dim, int num_heads,
                       bool causal = false, float dropout_p = 0.0f);
    ~MultiHeadAttention() override;

    Tensor* forward(NNContext* ctx, Tensor* x) override;   // x [B, T, E]
    void collect_parameters(std::vector<Tensor*>& out) override;

    Linear *q_proj, *k_proj, *v_proj, *out_proj;
    int embed_dim, num_heads, head_dim;
    bool causal;
    float dropout_p;
};

} // namespace nn
} // namespace ultraml
