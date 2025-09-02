#include "Bindings/pch.h"

NB_MODULE(light, m) { m.def("cuda_add", &Base::GetVersion, "Add two integers using a CUDA kernel"); }