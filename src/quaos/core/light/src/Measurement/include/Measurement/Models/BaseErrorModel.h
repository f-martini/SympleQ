#pragma once

#include "Measurement/Models/IErrorModel.h"

namespace Measurement {

class MEASUREMENT_API BaseErrorModel : public IErrorModel {
public:
    BaseErrorModel();

    // IErrorModel
    double ComputeError() override;
};

}  // namespace Measurement