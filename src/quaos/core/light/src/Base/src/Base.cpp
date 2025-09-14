#include "Base/pch.h"
#include "Base/Base.h"

namespace Base {

const int GetVersion() {
    int mkl_threads = mkl_get_max_threads();  // Suppress unused variable warning
    return mkl_threads;
}

}  // namespace Base