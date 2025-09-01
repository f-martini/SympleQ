#pragma once

namespace CudaProxy {

// Define CUDA function signatures you care about
typedef int (*CudaGetDeviceCount_t)(int*);
typedef int (*CudaMalloc_t)(void**, size_t);
typedef int (*CudaFree_t)(void*);

class CudaLoader {
   private:
    HMODULE hCuda = nullptr;

   public:
    // Function pointers
    CudaGetDeviceCount_t cudaGetDeviceCount = nullptr;
    CudaMalloc_t cudaMalloc = nullptr;
    CudaFree_t cudaFree = nullptr;

    bool loaded = false;

    CudaLoader();

    ~CudaLoader();
};

}  // namespace CudaProxy