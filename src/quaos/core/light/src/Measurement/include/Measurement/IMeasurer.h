#pragma once

namespace Measurement {

struct MEASUREMENT_API MeasureResult {
    std::vector<std::complex<double>> measures;
};

class MEASUREMENT_API IMeasurer {
public:
    virtual ~IMeasurer() = default;
    virtual MeasureResult Measure(uint64_t shots) = 0;
};

}  // namespace Measurement