#pragma once
#include <cmath>
#include <core/tensor/ops/base/UnaryElementWiseOp.hpp>

template <typename T> class ExpOp : public UnaryElementwiseOp<T> {
  protected:
    T forward_single(const T& x) const override { return std::exp(x); }
    T backward_single(const T& x, const T& y,
                      const T& grad_output) const override {
        return std::exp(x) * grad_output;
    }

  public:
    ExpOp(TensorPtr<T> input) : UnaryElementwiseOp<T>(input) {}
};
