// IntelliSense-only declaration for CUDA kernel launch syntax.
// This file is not included in the build.
#if defined(__INTELLISENSE__) && !defined(__CUDACC__)
#include <cuda_runtime.h>
extern "C" unsigned __cudaPushCallConfiguration(
    dim3 grid, dim3 block, size_t sharedMemory = 0, void* stream = nullptr);
#endif
