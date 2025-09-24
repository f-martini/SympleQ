#pragma once

#include "Measurement/IMeasurer.h"

namespace Measurement {

struct MEASUREMENT_API Pauli {
    uint64_t x;
    uint64_t z;
    double phase;
};

using PauliString = std::vector<Pauli>;
using PauliSum = std::vector<PauliString>;

struct AquireConfig;

class MEASUREMENT_API Aquire : public IMeasurer {
public:
    Aquire(PauliSum const& hamiltonian, std::vector<std::complex<double>> const& psi, AquireConfig const& config);

    // IMeasurer
    MeasureResult Measure(uint64_t shots) override;

private:
    PauliSum hamiltonian;
};

}  // namespace Measurement