// Host-side stub of the CUDA runtime for syntax/type-checking on machines
// without the CUDA toolkit. Signatures mirror the public API closely enough
// to type-check UltraML sources; never link against this.
#pragma once

#include <cstddef>
#include <cmath>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__
#define __constant__
#define __restrict__

struct uint3 { unsigned int x, y, z; };
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
};

// Builtin kernel variables (only referenced, never meaningfully evaluated).
extern uint3 threadIdx, blockIdx;
extern dim3  blockDim, gridDim;

inline void __syncthreads() {}
inline float atomicAdd(float* address, float val) { float o = *address; *address += val; return o; }
inline int   atomicAdd(int* address, int val)     { int o = *address; *address += val; return o; }
inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }

typedef enum { cudaSuccess = 0, cudaErrorUnknown = 999 } cudaError_t;

typedef enum {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
} cudaMemcpyKind;

cudaError_t cudaMalloc(void** devPtr, size_t size);
template <class T>
cudaError_t cudaMalloc(T** devPtr, size_t size) {
    return cudaMalloc((void**)devPtr, size);
}
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count,
                       cudaMemcpyKind kind);
cudaError_t cudaDeviceSynchronize(void);
const char* cudaGetErrorString(cudaError_t error);
