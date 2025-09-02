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
        auto gradA = TensorPtr<T>::create(A->getShape());
        auto gradB = TensorPtr<T>::create(B->getShape());

        auto A_b = (A->getShape() == this->output->getShape())
                       ? A
                       : A->broadcast_to(this->output->getShape());
        auto B_b = (B->getShape() == this->output->getShape())
                       ? B
                       : B->broadcast_to(this->output->getShape());

        std::vector<int> idx(this->output->getShape().size(), 0);
        do {
            auto [gA, gB] =
                backward_single((*A_b)(idx), (*B_b)(idx), (*grad_out)(idx));
            (*gradA)(idx) = gA;
            (*gradB)(idx) = gB;
        } while (Tensor<T>::incrementIndex(idx, this->output->getShape()));

        A->addGrad(gradA);
        B->addGrad(gradB);
    }
};
} // namespace FosterML
