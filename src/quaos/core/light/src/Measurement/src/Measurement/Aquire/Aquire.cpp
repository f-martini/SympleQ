#include "Measurement/pch.h"

#include "Measurement/Aquire/Aquire.h"

namespace Measurement {

Aquire::Aquire(PauliSum const& hamiltonian, std::vector<std::complex<double>> const& psi, AquireConfig const& config)
    : hamiltonian(hamiltonian) {}

MeasureResult Aquire::Measure(uint64_t shots) { return {}; }

}  // namespace Measurement
