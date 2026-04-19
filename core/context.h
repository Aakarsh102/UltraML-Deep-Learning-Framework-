#pragma once

#include <cublas_v2.h>
#include <cudnn.h>

namespace ultraml {

struct NNContext {
  cublasHandle_t cublas_handle;
  cudnnHandle_t cudnn_handle;
};

extern NNContext *create_context();
extern void destroy_context(NNContext *ctx);

} // namespace ultraml
