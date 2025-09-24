#pragma once

namespace Measurement {

class IErrorModel {
public:
    virtual ~IErrorModel() = default;

    virtual double ComputeError() = 0;
};

}  // namespace Measurement