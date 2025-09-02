#pragma once
#include <cmath>
#include <core/tensor/ops/base/UnaryElementWiseOp.hpp>

template <typename T> class LogOp : public UnaryElementwiseOp<T> {
  protected:
    T forward_single(const T& x) const override { return std::log(x); }
    T backward_single(const T& x, const T& y,
                      const T& grad_output) const override {
        return grad_output / x; // d/dx log(x) = 1/x
    }

  public:
    LogOp(TensorPtr<T> input) : UnaryElementwiseOp<T>(input) {}
};
