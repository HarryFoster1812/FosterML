#pragma once
#include <core/tensor/ops/base/BinaryElementWiseOp.hpp>
#include <limits>

namespace FosterML {
template <typename T> class DivOp : public BinaryElementwiseOp<T> {
  protected:
    T forward_single(const T& a, const T& b) const override {
        if (b == T(0)) {
            throw std::runtime_error("Division by zero in DivOp");
        }
        return a / b;
    }

    std::pair<T, T> backward_single(const T& a, const T& b,
                                    const T& grad_output) const override {
        // for division C = A/B
        // \partial C w.r.t A = 1/B
        // \partial C w.r.t B = -A/B^2
        const T eps = std::numeric_limits<T>::epsilon();
        const T denom = (b * b) + eps; // avoid div-by-zero
        return {grad_output / b, -grad_output * a / denom};
    }

  public:
    DivOp(TensorPtr<T> A, TensorPtr<T> B) : BinaryElementwiseOp<T>(A, B) {}
};
} // namespace FosterML
