#include "Measurement/pch.h"

#include "Measurement/Models/BaseNoiseModel.h"

namespace Measurement {

BaseNoiseModel::BaseNoiseModel() {}

double BaseNoiseModel::ComputeNoise() { return 0; }

}  // namespace Measurement