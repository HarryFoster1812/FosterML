#include <unordered_set>

template <typename T> class AutoDiffEngine {
public:
  void backward(Tensor<T> &output) {
    std::vector<std::shared_ptr<const Tensor<T>>> topo_order;
    std::unordered_set<const Tensor<T> *> visited;
    dfs(output.getSharedPtr(), visited, topo_order);

    output.setGradOnes();
    // it = input tensor
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {

      if ((*it)->requiresGrad() && (*it)->backwardsFunction) {
        std::cout << "GRADIENT INPUT:" << std::endl;
        (*it)->getGrad().print();
        (*it)->backwardsFunction(**it);
      }
    }
  }

private:
  void dfs(std::shared_ptr<const Tensor<T>> node,
           std::unordered_set<const Tensor<T> *> &visited,
           std::vector<std::shared_ptr<const Tensor<T>>> &order) {
    if (visited.count(node.get()))
      return;
    visited.insert(node.get());

    for (const auto &parent : node->parents) {
      dfs(parent, visited, order);
    }

    order.push_back(node);
  }
};
