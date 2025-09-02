#pragma once
#include <cmath>
#include <core/tensor/ops/base/UnaryElementWiseOp.hpp>

namespace FosterML {
template <typename T> class TanhOp : public UnaryElementwiseOp<T> {
  protected:
    T forward_single(const T& x) const override { return std::tanh(x); }
    T backward_single(const T& x, const T& y,
                      const T& grad_output) const override {
        // partial z / partial x = 1-tanh^2
        return grad_output * (1 - std::pow(y, 2));
    }

  public:
    TanhOp(TensorPtr<T> input) : UnaryElementwiseOp<T>(input) {}
};
} // namespace FosterML
