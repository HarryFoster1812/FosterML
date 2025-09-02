#pragma once
#include <core/tensor/ops/base/UnaryElementWiseOp.hpp>

template <typename T> class NegateOp : public UnaryElementwiseOp<T> {
  protected:
    T forward_single(const T& x) const override { return -x; }
    T backward_single(const T& x, const T& y,
                      const T& grad_output) const override {
        return -grad_output;
    }

  public:
    NegateOp(TensorPtr<T> input) : UnaryElementwiseOp<T>(input) {}
};
