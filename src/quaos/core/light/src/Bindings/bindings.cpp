#include "Bindings/pch.h"

NB_MODULE(light, m) {
    m.def("dot", &Base::GetVersion, "Perform matrix dot product");
    m.def("kron", &Base::GetVersion, "Perform Kroneker product");
    m.def("test", &Base::GetVersion, "Perform Kroneker product");
}