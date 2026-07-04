// Minimal CUDA error checking. Wrap every cudaMalloc / cudaMemcpy / cudaMemset
// and check cudaGetLastError() after each kernel launch — an unchecked CUDA
// call fails silently and hands back garbage.
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err_));                                 \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

// Check a kernel launch (config errors are reported synchronously here; async
// execution errors surface at the next synchronizing copy). Cheap — no sync.
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())
