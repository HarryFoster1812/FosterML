#pragma once
#include <cmath>
#include <core/tensor/ops/base/UnaryElementWiseOp.hpp>

namespace FosterML {
template <typename T> class SqrtOp : public UnaryElementwiseOp<T> {
  protected:
    T forward_single(const T& x) const override { return std::sqrt(x); }
    T backward_single(const T& x, const T& y,
                      const T& grad_output) const override {
        // partial z / partial x = 1/(2*sqrt(x))
        return grad_output / (2 * y);
    }

  public:
    SqrtOp(TensorPtr<T> input) : UnaryElementwiseOp<T>(input) {}
};
} // namespace FosterML
