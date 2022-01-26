#pragma once

#include "type_defs.h"

namespace warpctc {

static gpuError_t memcpy_d2h_async(void *dst, const void *src, size_t bytes, GPUstream stream) {
    gpuError_t status;
#ifdef __HIPCC__
    status = hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream);
#else
    status = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
#endif
    return status;
}

static gpuError_t memcpy_h2d_async(void *dst, const void *src, size_t bytes, GPUstream stream) {
    gpuError_t status;
#ifdef __HIPCC__
    status = hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream);
#else
    status = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
#endif
    return status;
}

static gpuError_t synchronize(GPUstream stream) {
    gpuError_t status;
#ifdef __HIPCC__
    status = hipStreamSynchronize(stream);
#else
    status = cudaStreamSynchronize(stream);
#endif
    return status;
}

static gpuError_t memcpy_d2h_sync(void *dst, const void *src, size_t bytes, GPUstream stream) {
    gpuError_t status = memcpy_d2h_async(dst, src, bytes, stream);
    if (status != gpuSuccess) {
        return status;
    }
    return synchronize(stream);
}

}  // namespace warpctc
