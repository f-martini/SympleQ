#pragma once

#include "Measurement/Models/BaseErrorModel.h"
#include "Measurement/Models/BaseNoiseModel.h"

namespace Measurement {

enum class AllocationMode {
    Undefined = 0,
    Set,
};

enum class DiagnosticMode {
    Undefined = 0,
    Zero,
};

struct MEASUREMENT_API AquireConfig {
    bool general_commutation = false;
    bool true_values = true;
    std::uint64_t N_chain = 8;
    std::uint64_t N_mcmc = 500;
    std::uint64_t N_mcmc_max = 2001;
    double mcmc_shot_scale = 1 / 10000;
    AllocationMode allocation_mode = AllocationMode::Undefined;
    DiagnosticMode diagnostic_mode = DiagnosticMode::Undefined;
    BaseNoiseModel noise_probability_function;
    BaseErrorModel error_function;
};

}  // namespace Measurement