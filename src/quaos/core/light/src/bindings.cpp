#include <nanobind/nanobind.h>
#include <>

#ifdef _WIN32
#include <windows.h>
#endif

void load_cuda_dll()
{
#ifdef _WIN32

    HMODULE hMod = nullptr;
    const char *cuda_path_c = std::getenv("CUDA_PATH");
    // You can try several known CUDA versions or query environment variables
    const char *cuda_paths[] = {
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin\\cudart64_120.dll",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin\\cudart64_118.dll",
        nullptr};

    for (const char **p = cuda_paths; *p != nullptr; ++p)
    {
        if (hMod = LoadLibraryA(*p); hMod)
            break; // Successfully loaded
    }

    if (!hMod)
        exit(-1);
#else

#endif
}

NB_MODULE(light, m)
{
    m.def("cuda_add", &LightMath::cuda_add, "Add two integers using a CUDA kernel");
}