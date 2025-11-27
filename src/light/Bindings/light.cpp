#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(light, m)
{
    m.def("test_import", []()
          { return "Light module imported successfully!"; }, "Test function to verify module import");
}