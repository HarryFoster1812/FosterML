#pragma once
#include <unordered_set>

template <typename T> class AutoDiffEngine {
public:
  void backward(TensorPtr<T> output) {
    std::vector<std::shared_ptr<Tensor<T>>> topo_order;
    std::unordered_set<Tensor<T> *> visited;
    dfs(output.get_shared_ptr(), visited, topo_order);

    output->setGradOnes();
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
      if ((*it)->requiresGrad() && (*it)->backwardsFunction) {
        (*it)->backwardsFunction(**it);
      }
    }
  }

private:
  void dfs(std::shared_ptr<Tensor<T>> node,
           std::unordered_set<Tensor<T> *> &visited,
           std::vector<std::shared_ptr<Tensor<T>>> &order) {
    if (visited.count(node.get()))
      return;
    visited.insert(node.get());

    for (const auto &parent : node->parents) {
      dfs(parent, visited, order);
    }

    order.push_back(node);
  }
};
