#include "Bindings/pch.h"

NB_MODULE(light, m) { m.def("cuda_add", &LightMath::cuda_add, "Add two integers using a CUDA kernel"); }