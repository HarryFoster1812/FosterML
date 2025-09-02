#pragma once
#include <cmath>
#include <core/tensor/ops/base/UnaryElementWiseOp.hpp>

template <typename T> class AbsOp : public UnaryElementwiseOp<T> {
  protected:
    T forward_single(const T& x) const override { return std::abs(x); }
    T backward_single(const T& x, const T& y,
                      const T& grad_output) const override {
        return -grad_output;
    }

  public:
    AbsOp(TensorPtr<T> input) : UnaryElementwiseOp<T>(input) {}
};
