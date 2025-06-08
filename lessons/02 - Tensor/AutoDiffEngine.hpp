#include <unordered_set>

template <typename T> class AutoDiffEngine {
public:
  void backward(Tensor<T> *output) {
    std::vector<Tensor<T> *> topo_order;
    std::unordered_set<Tensor<T> *> visited;
    dfs(output, visited, topo_order);
    if (output->getGrad().getData().empty()) {
      if (output->getData().size() != 1) {
        throw std::runtime_error(
            "Output gradient initialization requires scalar output");
      }
      output->getGrad().setData({static_cast<T>(1)});
    }
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
      if ((*it)->requiresGrad() && (*it)->backwardsFunction) {
        (*it)->backwardsFunction(**it);
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
