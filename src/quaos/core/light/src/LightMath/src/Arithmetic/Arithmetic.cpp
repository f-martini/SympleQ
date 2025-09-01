#include "LightMath/pch.h"

#include "LightMath/Arithmetic.h"

namespace LightMath {

int cuda_add(int a, int b) {
    int *result_host = new int;
    launch_add_kernel(a, b, result_host);
    int result = *result_host;
    delete result_host;
    return result;
}

}  // namespace LightMath
