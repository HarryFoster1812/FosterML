#include <Tensor.hpp>
#include <unordered_set>

template <typename T> class AutoDiffEngine {
public:
  void backward(Tensor<T> *output) {
    // Step 1: Topological sort
    std::vector<Tensor<T> *> topo_order;
    std::unordered_set<Tensor<T> *> visited;
    dfs(output, visited, topo_order);

    // Step 2: Initialize gradient at output
    if (output->getGrad().getData().empty()) {

      output->getGrad().setData(
          std::vector<T>(output->getData().size(), static_cast<T>(1)));
    }

    // Step 3: Backward pass in reverse topological order
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
      if ((*it)->requires_grad && (*it)->backward_fn) {
        (*it)->backward_fn();
      }
    }
  }

private:
  void dfs(Tensor<T> *node, std::unordered_set<Tensor<T> *> &visited,
           std::vector<Tensor<T> *> &order) {
    if (visited.count(node))
      return;
    visited.insert(node);

    for (Tensor<T> *parent : node->getParents()) {
      dfs(parent, visited, order);
    }

    order.push_back(node);
  }
};
