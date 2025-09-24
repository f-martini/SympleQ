#include "Measurement/pch.h"

#include "Measurement/Models/BaseErrorModel.h"

namespace Measurement {

BaseErrorModel::BaseErrorModel() {}

double BaseErrorModel::ComputeError() { return 0; }

}  // namespace Measurement