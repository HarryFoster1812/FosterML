#pragma once
#include <core/tensor/ops/base/BinaryElementWiseOp.hpp>

namespace FosterML {
template <typename T> class SubOp : public BinaryElementwiseOp<T> {
  protected:
    T forward_single(const T& a, const T& b) const override { return a + b; }

    std::pair<T, T> backward_single(const T& a, const T& b,
                                    const T& grad_output) const override {
        return {grad_output, -grad_output};
    }

  public:
    SubOp(TensorPtr<T> A, TensorPtr<T> B) : BinaryElementwiseOp<T>(A, B) {}
};
} // namespace FosterML
