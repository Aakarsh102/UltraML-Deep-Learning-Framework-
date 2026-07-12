#include "autograd.h"
#include "../core/macros.h"

#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <utility>

namespace ultraml {
namespace autograd {

namespace {

// ---- engine kernels -------------------------------------------------
__global__ void k_fill_ones(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = 1.0f;
}

__global__ void k_accumulate(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

void launch_fill_ones(float* data, int n) {
    int t = ULTRAML_CUDA_BLOCK, blk = (n + t - 1) / t;
    k_fill_ones<<<blk, t>>>(data, n);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_accumulate(float* dst, const float* src, int n) {
    int t = ULTRAML_CUDA_BLOCK, blk = (n + t - 1) / t;
    k_accumulate<<<blk, t>>>(dst, src, n);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

// ---- tape state ------------------------------------------------------
// One global tape; not thread-safe. Nodes are appended during forward and
// freed by clear(); owned tensors are the intermediates created by ops::.
bool                       g_grad_enabled = true;
std::vector<AutogradNode*> g_nodes;
std::vector<Tensor*>       g_owned;

// Ensures t->grad exists without clobbering an existing (possibly
// accumulated) gradient. alloc_grad() zero-fills unconditionally, so it is
// only safe to call when the buffer is absent.
void ensure_grad(Tensor* t) {
    if (!t->grad) alloc_grad(t);
}

} // namespace

bool grad_enabled() { return g_grad_enabled; }
void set_grad_enabled(bool enabled) { g_grad_enabled = enabled; }

NoGradGuard::NoGradGuard() : prev_(g_grad_enabled) { g_grad_enabled = false; }
NoGradGuard::~NoGradGuard() { g_grad_enabled = prev_; }

Tensor* own(Tensor* t) {
    g_owned.push_back(t);
    return t;
}

void record(const char* op_name,
            std::vector<Tensor*> inputs,
            Tensor* output,
            BackwardFn backward_fn) {
    if (!g_grad_enabled) return;

    bool any_requires = false;
    for (Tensor* in : inputs) {
        if (in && in->requires_grad) { any_requires = true; break; }
    }
    if (!any_requires) return;

    AutogradNode* node = new AutogradNode();
    node->op_name     = op_name;
    node->inputs      = std::move(inputs);
    node->output      = output;
    node->backward_fn = std::move(backward_fn);
    g_nodes.push_back(node);

    output->grad_fn       = node;
    output->requires_grad = true;
}

void backward(NNContext* ctx, Tensor* loss) {
    if (!loss->grad_fn) {
        fprintf(stderr,
                "autograd::backward: tensor has no grad_fn "
                "(nothing was recorded — is grad enabled and does some "
                "input have requires_grad?)\n");
        return;
    }

    // Seed d(loss)/d(loss) = 1.
    ensure_grad(loss);
    launch_fill_ones(loss->grad, loss->size);

    // Iterative post-order DFS over producer edges: `order` ends up with
    // producers before consumers and the loss node last.
    std::vector<AutogradNode*>              order;
    std::unordered_set<AutogradNode*>       visited;
    std::vector<std::pair<AutogradNode*, bool>> stack;
    stack.push_back({loss->grad_fn, false});

    while (!stack.empty()) {
        auto [node, children_done] = stack.back();
        stack.pop_back();
        if (children_done) { order.push_back(node); continue; }
        if (visited.count(node)) continue;
        visited.insert(node);
        stack.push_back({node, true});
        for (Tensor* in : node->inputs) {
            if (in && in->grad_fn && !visited.count(in->grad_fn)) {
                stack.push_back({in->grad_fn, false});
            }
        }
    }

    // Execute consumers before producers so every node sees its complete
    // output gradient.
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        AutogradNode* node = *it;

        if (!node->output->grad) {
            // Unreachable in a well-formed graph; treat as zero gradient.
            ensure_grad(node->output);
        }

        // Shallow view: same shape, data = accumulated output gradient.
        Tensor grad_out_view   = *node->output;
        grad_out_view.data     = node->output->grad;
        grad_out_view.grad     = nullptr;
        grad_out_view.grad_fn  = nullptr;

        // Zero-filled scratch per differentiable input; the closure writes
        // gradients there, then we accumulate so shared inputs sum up.
        std::vector<Tensor*> grad_inputs(node->inputs.size(), nullptr);
        for (size_t i = 0; i < node->inputs.size(); ++i) {
            Tensor* in = node->inputs[i];
            if (in && in->requires_grad) {
                grad_inputs[i] = create_tensor(in->shape, in->ndim, false);
                zero_tensor(grad_inputs[i]);
            }
        }

        node->backward_fn(ctx, &grad_out_view, grad_inputs);

        for (size_t i = 0; i < node->inputs.size(); ++i) {
            if (!grad_inputs[i]) continue;
            Tensor* in = node->inputs[i];
            ensure_grad(in);
            launch_accumulate(in->grad, grad_inputs[i]->data, in->size);
            free_tensor(grad_inputs[i]);
        }
    }
}

void clear() {
    for (AutogradNode* node : g_nodes) delete node;
    g_nodes.clear();
    for (Tensor* t : g_owned) free_tensor(t);
    g_owned.clear();
}

size_t num_nodes() { return g_nodes.size(); }

} // namespace autograd
} // namespace ultraml
