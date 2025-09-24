#pragma once

#include "Measurement/Models/INoiseModel.h"

namespace Measurement {

class MEASUREMENT_API BaseNoiseModel : public INoiseModel {
public:
    BaseNoiseModel();

    // INoiseModel
    double ComputeNoise() override;
};

}  // namespace Measurement