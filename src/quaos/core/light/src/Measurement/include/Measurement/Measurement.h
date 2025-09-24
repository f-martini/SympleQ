#pragma once

#include "Measurement/MeasurementExport.h"
#include "Measurement/Aquire/AquireConfig.h"
#include "Measurement/Aquire/Aquire.h"
#include "Measurement/Models/BaseErrorModel.h"
#include "Measurement/Models/BaseNoiseModel.h"

namespace Measurement {

constexpr auto VERSION = "1.0.0";

MEASUREMENT_API const char* GetVersion();

}  // namespace Measurement

#undef MEASUREMENT_API