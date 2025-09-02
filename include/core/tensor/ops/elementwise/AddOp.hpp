#pragma once
#include <core/tensor/ops/base/BinaryElementWiseOp.hpp>

template <typename T> class AddOp : public BinaryElementwiseOp<T> {
  protected:
    T forward_single(const T& a, const T& b) const override { return a + b; }

    std::pair<T, T> backward_single(const T& a, const T& b,
                                    const T& grad_output) const override {
        return {grad_output, grad_output};
    }

  public:
    AddOp(TensorPtr<T> A, TensorPtr<T> B) : BinaryElementwiseOp<T>(A, B) {}
};
