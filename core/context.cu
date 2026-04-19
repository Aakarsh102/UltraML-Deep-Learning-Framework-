#include "context.h"
#include "macros.h"

#include <cstdlib>

namespace ultraml {

NNContext* create_context() {
    NNContext* ctx = (NNContext*)std::malloc(sizeof(NNContext));
    ULTRAML_CUBLAS_CHECK(cublasCreate(&ctx->cublas_handle));
    ULTRAML_CUDNN_CHECK(cudnnCreate(&ctx->cudnn_handle));
    return ctx;
}

void destroy_context(NNContext* ctx) {
    if (!ctx) return;
    ULTRAML_CUBLAS_CHECK(cublasDestroy(ctx->cublas_handle));
    ULTRAML_CUDNN_CHECK(cudnnDestroy(ctx->cudnn_handle));
    std::free(ctx);
}

} // namespace ultraml
