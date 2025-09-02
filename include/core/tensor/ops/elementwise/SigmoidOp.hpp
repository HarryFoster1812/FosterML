#pragma once
#include <cmath>
#include <core/tensor/ops/base/UnaryElementWiseOp.hpp>

namespace FosterML {
template <typename T> class SigmoidOp : public UnaryElementwiseOp<T> {
  protected:
    T forward_single(const T& x) const override {
        return (T(1)) / (T(1) + std::exp(-x));
    }
    T backward_single(const T& x, const T& y,
                      const T& grad_output) const override {
        return grad_output * (y / (1 - y));
    }

  public:
    SigmoidOp(TensorPtr<T> input) : UnaryElementwiseOp<T>(input) {}
};
} // namespace FosterML
