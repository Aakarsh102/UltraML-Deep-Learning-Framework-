#include "layers.h"
#include "../core/macros.h"

namespace ultraml {

namespace {

// Stateless counter-based RNG (splitmix64 finalizer). The value depends only
// on (seed, index), so a mask is reproducible without storing RNG state.
__device__ __forceinline__ unsigned int mix_hash(unsigned long long seed,
                                                 unsigned int idx) {
    unsigned long long z = seed + 0x9E3779B97F4A7C15ULL * (idx + 1ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z =  z ^ (z >> 31);
    return (unsigned int)(z & 0xFFFFFFFFULL);
}

__global__ void k_dropout_mask(float* mask, int n, float p, float scale,
                               unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float r = (float)mix_hash(seed, (unsigned int)i) / 4294967295.0f;
        mask[i] = (r < p) ? 0.0f : scale;
    }
}

} // namespace

void dropout_make_mask(Tensor* mask, float p, unsigned long long seed) {
    float scale = (p < 1.0f) ? 1.0f / (1.0f - p) : 0.0f;
    int t = ULTRAML_CUDA_BLOCK;
    int blk = (mask->size + t - 1) / t;
    k_dropout_mask<<<blk, t>>>(mask->data, mask->size, p, scale, seed);
    ULTRAML_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace ultraml
