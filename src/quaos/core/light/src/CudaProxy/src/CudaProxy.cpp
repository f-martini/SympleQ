#include "CudaProxy/pch.h"

#include "CudaProxy/CudaProxy.h"

namespace CudaProxy {

CudaLoader::CudaLoader() {
    // Try loading the CUDA runtime DLL
    hCuda = LoadLibraryA("cudart64_120.dll");  // Example: CUDA 12.0 runtime DLL
    if (!hCuda) {
        std::cerr << "CUDA runtime not found. Falling back to CPU.\n";
        return;
    }

    // Resolve functions
    cudaGetDeviceCount = (CudaGetDeviceCount_t)GetProcAddress(hCuda, "cudaGetDeviceCount");
    cudaMalloc = (CudaMalloc_t)GetProcAddress(hCuda, "cudaMalloc");
    cudaFree = (CudaFree_t)GetProcAddress(hCuda, "cudaFree");

    // Verify all required functions exist
    if (!cudaGetDeviceCount || !cudaMalloc || !cudaFree) {
        std::cerr << "Failed to load required CUDA functions.\n";
        FreeLibrary(hCuda);
        hCuda = nullptr;
        return;
    }

    loaded = true;
    std::cout << "CUDA runtime successfully loaded.\n";
}

CudaLoader::~CudaLoader() {
    if (hCuda) {
        FreeLibrary(hCuda);
    }
}

}  // namespace CudaProxy