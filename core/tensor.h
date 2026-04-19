#pragma once

#include <cstddef>

namespace ultraml {

// Forward-declared so future autograd can attach a graph node without
// changing Tensor's ABI. All current ops take their "saved" inputs as
// explicit arguments to backward(), which is the same contract autograd
// will use when it records them inside an AutogradNode.
struct AutogradNode;

struct Tensor {
    float* data;          // device pointer
    int*   shape;         // host-side shape array, length ndim
    int    ndim;
    int    size;          // product of shape

    // Gradient buffer; nullptr unless alloc_grad() has been called.
    float* grad;
    bool   requires_grad;

    // Opaque handle for the future autograd graph. Always nullptr today.
    AutogradNode* grad_fn;
};

// Allocation --------------------------------------------------------------
Tensor* create_tensor(const int* shape, int ndim, bool requires_grad = false);
void    free_tensor(Tensor* t);

// Data movement -----------------------------------------------------------
void fill_tensor(Tensor* t, float value);
void zero_tensor(Tensor* t);
void copy_tensor(Tensor* dst, const Tensor* src);              // device->device
void copy_from_host(Tensor* dst, const float* host_data);      // host->device
void copy_to_host(const Tensor* src, float* host_data);        // device->host

// Gradient helpers --------------------------------------------------------
// Allocates t->grad lazily (same size as t->data) and zero-fills it.
void alloc_grad(Tensor* t);
void zero_grad(Tensor* t);
void free_grad(Tensor* t);

} // namespace ultraml
