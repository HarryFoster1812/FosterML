#pragma once
#include "OpNode.hpp"

template <typename T> class UnaryElementwiseOp : public OpNode<T> {
  public:
    UnaryElementwiseOp(TensorPtr<T> input)
        : OpNode<T>({input}, input->getShape(), input->requiresGrad()) {}

  protected:
    // Each subclass implements these
    virtual T forward_single(const T& x) const = 0;
    virtual T backward_single(const T& x, const T& y,
                              const T& grad_output) const = 0;

  public:
    void forward() override {
        const auto& in_data = this->inputs[0]->getData();
        auto& out_data = this->output->getData();
        for (size_t i = 0; i < in_data.size(); ++i)
            (*out_data)[i] = forward_single(in_data[i]);
    }

    void backward() override {
        if (!this->inputs[0]->requiresGrad())
            return;

        const auto& grad_out = this->output->getGrad()->getData();
        auto grad_in = this->inputs[0]->getGrad();
        const auto& in_data = this->inputs[0]->getData();
        const auto& out_data = this->output->getData();

        for (size_t i = 0; i < grad_out->size(); ++i) {
            (*grad_in->getData())[i] +=
                backward_single((*in_data)[i], (*out_data)[i], (*grad_out)[i]);
        }
    }
};
