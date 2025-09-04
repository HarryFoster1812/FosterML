#pragma once
#include "OpNode.hpp"

namespace FosterML {

template <typename T> class BinaryElementwiseOp : public OpNode<T> {
  public:
    BinaryElementwiseOp(TensorPtr<T> A, TensorPtr<T> B)
        : OpNode<T>({A, B}, Tensor<T>::infer_broadcast_shape(A, B),
                    A->requiresGrad() || B->requiresGrad()) {}

  protected:
    virtual T forward_single(const T& a, const T& b) const = 0;
    virtual std::pair<T, T> backward_single(const T& a, const T& b,
                                            const T& grad_output) const = 0;

  public:
    void forward() override {
        auto A = this->inputs[0];
        auto B = this->inputs[1];
        auto A_b = (A->getShape() == this->output->getShape())
                       ? A
                       : A->broadcast_to(this->output->getShape());
        auto B_b = (B->getShape() == this->output->getShape())
                       ? B
                       : B->broadcast_to(this->output->getShape());

        std::vector<int> idx(this->output->getShape().size(), 0);
        do {
            (*this->output)(idx) = forward_single((*A_b)(idx), (*B_b)(idx));
        } while (Tensor<T>::incrementIndex(idx, this->output->getShape()));
    }

    void backward() override {
        auto grad_out = this->output->getGrad();
        auto A = this->inputs[0];
        auto B = this->inputs[1];

        const auto& shapeA = A->getShape();
        const auto& shapeB = B->getShape();
        const auto& outShape = this->output->getShape();

        // Broadcast inputs to output shape
        auto A_b = (shapeA == outShape) ? A : A->broadcast_to(outShape);
        auto B_b = (shapeB == outShape) ? B : B->broadcast_to(outShape);

        // Gradients in broadcasted space
        auto gradA_b = TensorPtr<T>::create(outShape, false);
        auto gradB_b = TensorPtr<T>::create(outShape, false);

        std::vector<int> idx(outShape.size(), 0);
        do {
            // Get partial derivatives from your custom backward rule
            auto [gA, gB] =
                backward_single((*A_b)(idx), (*B_b)(idx), (*grad_out)(idx));

            (*gradA_b)(idx) = gA;
            (*gradB_b)(idx) = gB;
        } while (Tensor<T>::incrementIndex(idx, outShape));

        // Reduce back to original shapes if broadcasting happened
        auto gradA = Tensor<T>::sum_over_broadcasted_axes(gradA_b, shapeA);
        auto gradB = Tensor<T>::sum_over_broadcasted_axes(gradB_b, shapeB);

        if (A->requiresGrad())
            A->addGrad(gradA);
        if (B->requiresGrad())
            B->addGrad(gradB);
    }
};
} // namespace FosterML
