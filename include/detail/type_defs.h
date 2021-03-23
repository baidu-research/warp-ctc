#pragma once

#if (defined(__CUDACC__) || defined(__HIPCC__))

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#ifdef __HIPCC__
#define gpuSuccess hipSuccess
using gpuStream_t = hipStream_t;
using gpuError_t = hipError_t;
using gpuEvent_t = hipEvent_t;
#else
#define gpuSuccess cudaSuccess
using gpuStream_t = cudaStream_t;
using gpuError_t = cudaError_t;
using gpuEvent_t = cudaEvent_t;
#endif

#endif
