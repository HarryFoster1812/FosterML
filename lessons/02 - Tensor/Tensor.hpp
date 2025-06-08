#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

/*
NOTE:
ONCE THIS IS FINISHED AND WORKING DO NOT TOUCH IT - WHATEVER THE PROBLEM IS WORK
AROUND IT. I APPOLOGISE IN ADVANCE FOR THE TERRIBLE CODE (THIS IS MY FIRST TIME
WORKING IN C++)

I DO UNDERSTAND THAT SHARED POINTERS ARE BETTER BECAUSE YOU DONT NEED TO
MANUALLY TRACK THE LIFETIME BUT GOD DANM THEY ARE ANNOYING TO DEAL WITH
 */

template <typename T> class AutoDiffEngine;
template <typename T>
class Tensor : public std::enable_shared_from_this<Tensor<T>> {
  friend class AutoDiffEngine<T>;

private:
  std::shared_ptr<std::vector<T>> data;
  std::vector<int> shape;
  std::vector<int> strides;
  std::unique_ptr<Tensor<T>> gradient;
  bool requires_gradient = false;
  std::vector<std::shared_ptr<const Tensor<T>>> parents;
  std::function<void(const Tensor<T> &)> backwardsFunction;
  bool is_shared = false; // Tracks if managed by shared_ptr

  void printRecursive(std::vector<int> &indices, int dim, int depth) const;

public:
  // Constructors
  Tensor(std::vector<int> shape_, bool requires_grad = false)
      : shape(shape_), requires_gradient(requires_grad) {
    computeStrides();
    int total_size = 1;
    for (int dim : shape)
      total_size *= dim;
    data = std::make_shared<std::vector<T>>(total_size);
    if (requires_gradient)
      gradient = std::unique_ptr<Tensor<T>>(new Tensor<T>(shape_, false));
  }

  Tensor(const Tensor<T> &other)
      : data(other.data), shape(other.shape), strides(other.strides),
        requires_gradient(other.requires_gradient), parents(other.parents),
        backwardsFunction(other.backwardsFunction) {
    if (requires_gradient) {
      gradient = std::make_unique<Tensor<T>>(*other.gradient);
    }
  }
  void setShared(bool shared) { is_shared = shared; } // Set shared status
  bool isShared() const { return is_shared; }
  T &operator()(const std::initializer_list<int> &index_values);
  T &operator()(const std::vector<int> &index_values);
  const T &operator()(const std::vector<int> &index_values) const;

  const std::vector<int> &getShape() const { return shape; }
  const std::vector<T> &getData() const { return *data; }
  const Tensor<T> &getGrad() const { return *gradient; }

  // Add parent by shared_ptr
  void addParent(std::shared_ptr<const Tensor<T>> parent) {
    parents.push_back(parent); // Store raw pointer from shared_ptr
  }
  // Add parent by reference (requires that the tensor can provide shared_ptr to
  // itself)
  std::shared_ptr<const Tensor<T>> getSharedPtr() const {
    if (is_shared) {
      return this->shared_from_this();
    } else {
      // Create a new shared_ptr for non-shared tensors
      return std::make_shared<Tensor<T>>(*this);
    }
  }
  bool requiresGrad() const { return requires_gradient; }

  void zeroGrad() {
    if (requires_gradient && gradient != nullptr) {
      gradient->fill(static_cast<T>(0));
    }
  }

  void addGrad(const Tensor<T> &grad) const {
    if (requires_gradient) {
      std::cout << "Original Grad:" << std::endl;
      gradient->print();

      std::cout << "Grad to add:" << std::endl;
      grad.print();

      gradient->addInPlace(grad);
    }
  }

  void addInPlace(const Tensor<T> &other) {
    if (this->data->size() != other.data->size())
      throw std::runtime_error("Gradient size mismatch in addInPlace");
    for (size_t i = 0; i < data->size(); ++i) {
      (*data)[i] += (*other.data)[i];
    }
  }

  void setGradOnes() {
    if (requires_gradient && gradient != nullptr) {
      gradient->fill(static_cast<T>(1));
    }
  }

  void fill(T value) { std::fill((*data).begin(), (*data).end(), value); }

  void setData(const std::vector<T> &new_data) {
    if (!data) {
      throw std::runtime_error("Data pointer is null in setData");
    }
    if (new_data.size() != data->size()) {
      throw std::runtime_error("Size mismatch in setData");
    }
    *data = new_data; // Assign directly to avoid reallocating
  }

  void setStrides(const std::vector<int> &new_strides) {
    strides = new_strides;
  }

  void shareData(const std::shared_ptr<std::vector<T>> &shared_data) {
    data = shared_data;
  }

  void setBackwardFunction(std::function<void(const Tensor<T> &)> func) {
    backwardsFunction = func;
  }

  Tensor<T>(Tensor<T> &&) = default;
  Tensor<T> &operator=(Tensor<T> &&) = default;
  Tensor<T> &operator=(const Tensor<T> &) = default;
  ~Tensor() = default;

  // Core Arithmetic

  Tensor<T> add(const Tensor<T> &other) const;
  Tensor<T> add(const T &scalar) const;

  Tensor<T> subtract(const Tensor<T> &other) const;
  Tensor<T> subtract(const T &scalar) const;

  Tensor<T> multiply(const Tensor<T> &other) const;
  Tensor<T> multiply(const T &scalar) const;

  Tensor<T> divide(const Tensor<T> &other) const;
  Tensor<T> divide(const T &scalar) const;

  // Element Wise operations
  Tensor<T> negate() const;
  Tensor<T> abs() const;     // element-wise absolute value
  Tensor<T> sqrt() const;    // element-wise square root
  Tensor<T> exp() const;     // element-wise exponentiation
  Tensor<T> log() const;     // element-wise natural logarithm
  Tensor<T> sigmoid() const; // element-wise sigmoid
  Tensor<T> tanh() const;    // element-wise hyperbolic tangent
  Tensor<T> sign() const;

  void computeStrides() {
    strides.resize(shape.size());
    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= shape[i];
    }
  }

  Tensor<T> matrixmul(const Tensor<T> &other) const;

  Tensor<T> sum(const std::vector<int> &axis, bool keepdims) const;

  Tensor<T> broadcast_to(const std::vector<int> &new_shape,
                         bool matrix_mul = false) const;

  Tensor<T> transpose(int dim1, int dim2) const;
  static std::vector<int> infer_broadcast_shape(const Tensor<T> &tensorA,
                                                const Tensor<T> &tensorB);

  static std::vector<int> infer_broadcast_shape(const std::vector<int> &shapeA,
                                                const std::vector<int> &shapeB);

  static bool incrementIndex(std::vector<int> &index,
                             const std::vector<int> &shape) {
    for (int dim = shape.size() - 1; dim >= 0; --dim) {
      index[dim]++;
      if (index[dim] < shape[dim]) {
        return true; // successful increment
      }
      index[dim] = 0; // carry to next higher dimension
    }
    return false; // overflow, end of iteration
  }

  // Given an input gradient which has been broacasted it will output a tensor
  // with the same shape of the target which the gradient has been reduced to
  static Tensor<T> sum_over_broadcasted_axes(const Tensor<T> &gradient,
                                             const Tensor<T> &target);

  void unravel_index(int flat_index, std::vector<int> &indices_out) const;
  void print() const;

  // TO IMPLEMENT

  Tensor<T> mean(int dim) const;   // mean over a dimension
  Tensor<T> max(int dim) const;    // max over a dimension
  Tensor<T> min(int dim) const;    // min over a dimension
  Tensor<T> argmax(int dim) const; // index of max value along dim
  Tensor<T> argmin(int dim) const; // index of min value along dim

  // Element-Wise comparison
  Tensor<bool> equal(const Tensor<T> &other) const;
  Tensor<bool> greater(const Tensor<T> &other) const;
  Tensor<bool> less(const Tensor<T> &other) const;
  Tensor<bool> greater_equal(const Tensor<T> &other) const;
  Tensor<bool> less_equal(const Tensor<T> &other) const;

  Tensor<T> slice(const std::vector<int> &start_indices,
                  const std::vector<int> &sizes) const;
  Tensor<T> gather(int dim, const Tensor<int> &indices) const;

  // OPERATOR SYNTACTIC SUGAR WITH SMART MULTIPLICATION (AUTO MATRIX)

protected:
};

#include "Tensor.tpp"
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

/*
TODO:

abs
sqrt
exp
log
sigmoid
tanh
matmul
transpose
sum
mean
max
min
argmax
argmin
equal
greater
less
greater_equal
less_equal
broadcast_to
slice
gather
canBatchMultiply
batchMultiply

DONE:

Tensor
Tensor(Tensor<T>&&) = default
Tensor(const Tensor<T>&) = default
operator=
~Tensor
getShape
getData
getGrad
getParents
requiresGrad
zeroGrad
setData
setBackwardsFunction
computeStrides
add(const Tensor<T>&)
reshape
operator*
operator+
operator-
operator/
infer_broadcast_shape
add(const T&)
subtract(const Tensor<T>&)
subtract(const T&)
multiply(const Tensor<T>&)
multiply(const T&)
divide(const Tensor<T>&)
divide(const T&)
negate
incrementIndex
*/
