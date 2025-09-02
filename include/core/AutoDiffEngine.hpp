#pragma once
#include <core/tensor/Tensor.hpp>
#include <memory>
#include <unordered_set>
#include <vector>

namespace FosterML {

template <typename T> class AutoDiffEngine {
  public:
    void backward(TensorPtr<T> output) {
        std::vector<std::shared_ptr<Tensor<T>>> topo_order;
        std::unordered_set<Tensor<T>*> visited;
        buildTopo(output->getCreator(), visited, topo_order);

        output->setGradOnes(); // start backward

        // traverse in reverse topo order
        for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
            auto op_node = (*it)->getCreator();
            if (op_node) {
                op_node->backward();
            }
        }
    }

  private:
    void buildTopo(std::shared_ptr<OpNode<T>> node,
                   std::unordered_set<Tensor<T>*>& visited,
                   std::vector<std::shared_ptr<Tensor<T>>>& order) {
        if (!node)
            return;

        for (auto& input : node->getInputs()) {
            if (!visited.count(input.get())) {
                visited.insert(input.get());
                buildTopo(input->getCreator(), visited, order);
            }
        }

        order.push_back(node->getOutput());
    }
};

} // namespace FosterML
