#pragma once
#include <core/tensor/TensorPtr.hpp>
#include <memory>
#include <vector>

namespace FosterML {

template <typename T>
class OpNode : public std::enable_shared_from_this<OpNode<T>> {
  protected:
    std::vector<TensorPtr<T>> inputs;
    TensorPtr<T> output;

  public:
    OpNode(const std::vector<TensorPtr<T>>& inputs_,
           const std::vector<int>& outputShape, bool requiresGrad)
        : inputs(inputs_) {
        output = TensorPtr<T>::create(outputShape, requiresGrad);
        // link back so that the output knows its creator node
    }

    virtual ~OpNode() = default;

    // compute output tensor
    virtual void forward() = 0;

    // propagate gradient to inputs
    virtual void backward() = 0;

    TensorPtr<T> getOutput() const { return output; }

    void setOutput(TensorPtr<T> out) { output = out; }

    const std::vector<TensorPtr<T>>& getInputs() const { return inputs; }

    template <typename Derived, typename... Args>
    static std::shared_ptr<Derived> create(Args&&... args) {
        auto op = std::make_shared<Derived>(std::forward<Args>(args)...);

        if (op->getOutput()->requiresGrad())
            op->getOutput()->setCreator(op);

        return op;
    }
};
} // namespace FosterML
