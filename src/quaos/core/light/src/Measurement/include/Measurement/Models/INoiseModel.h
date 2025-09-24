#pragma once

namespace Measurement {

class INoiseModel {
public:
    virtual ~INoiseModel() = default;

    virtual double ComputeNoise() = 0;
};

}  // namespace Measurement