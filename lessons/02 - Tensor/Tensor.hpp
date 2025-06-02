#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

template <typename T> 
class Tensor : public std::enable_shared_from_this<Tensor<T>> {
  friend class AutoDiffEngine;

private:
  std::vector<T> data;
  std::vector<int> shape;
  std::vector<int> strides;
  std::unique_ptr<Tensor<T>> gradient;
  bool requires_gradient = false;

  std::vector<std::shared_ptr<Tensor<T>>>
      parents; // the parents that were used to compute the tensor
  std::function<void(Tensor<T> &)>
      backwardsFunction; // the function called to alter the parents gradients
  void printRecursive(std::vector<int> indices, int dim, int depth) const;

public:
  // Constructors
  Tensor(std::vector<int> shape_, bool requires_grad = false)
      : shape(shape_), requires_gradient(requires_grad) {
    computeStrides();
    int total_size = 1;
    for (int dim : shape)
      total_size *= dim;
    data.resize(total_size);
    if (requires_gradient)
            gradient = std::unique_ptr<Tensor<T>>(new Tensor<T>(shape_, false));
  }

  T &operator()(const std::initializer_list<int> &index_values) const;
  T &operator()(const std::vector<int> &index_values) const;

  const std::vector<int> &getShape() const { return shape; }
  const std::vector<T> &getData() const { return data; }
  const Tensor<T> &getGrad() const { return *gradient; }
  const std::vector<std::shared_ptr<Tensor<T>>> getParents() { return parents; }


    // Add parent by shared_ptr
    void addParent(const std::shared_ptr<Tensor<T>>& parent) {
        if (!parents_) {
            parents_ = std::make_shared<std::vector<std::weak_ptr<Tensor<T>>>>();
        }
        parents_->push_back(parent);
    }

    // Add parent by reference (requires that the tensor can provide shared_ptr to itself)
    void addParent(const Tensor<T>& parent) {
        addParent(parent.shared_from_this());
    }  bool requiresGrad() const { return requires_gradient; }

  void zeroGrad() {
    if (requires_gradient) {
      std::vector<T> gradientVector = gradient.getData();
      std::fill(gradientVector.begin(), gradientVector.end(),
                static_cast<T>(0));
    }
  }

 void addGrad(const Tensor<T>& grad){
        if(requires_gradient)
        gradient->add(grad);
    }

  void setData(const std::vector<T> &new_data) {
    if (new_data.size() != data.size())
      throw std::runtime_error("Size mismatch in setData");
    data = new_data;
  }

  void setBackwardsFunction(std::function<void(Tensor<T> &)> func) {
    backwardsFunction = func;
  }

  Tensor<T>(Tensor<T> &&) = default;
  Tensor<T>(const Tensor<T> &) = default;
  Tensor<T> &operator=(Tensor<T> &&) = default;
  Tensor<T> &operator=(const Tensor<T> &) = default;
  ~Tensor() = default;

  // Core Arithmetic

  Tensor<T> element_op() const;

  Tensor<T> add(const Tensor<T> &other) const;
  Tensor<T> add(const T &scalar) const;

  Tensor<T> subtract(const Tensor<T> &other) const;
  Tensor<T> subtract(const T &scalar) const;

  Tensor<T> multiply(const Tensor<T> &other) const;
  Tensor<T> multiply(const T &scalar) const;

  Tensor<T> divide(const Tensor<T> &other) const;
  Tensor<T> divide(const T &scalar) const;

  // Element Wise operations

  Tensor<T> negate() const;  // element-wise negation (-x)
  Tensor<T> abs() const;     // element-wise absolute value
  Tensor<T> sqrt() const;    // element-wise square root
  Tensor<T> exp() const;     // element-wise exponentiation
  Tensor<T> log() const;     // element-wise natural logarithm
  Tensor<T> sigmoid() const; // element-wise sigmoid
  Tensor<T> tanh() const;    // element-wise hyperbolic tangent

  // Linear Algrbra Operations
  Tensor<T>
  matmul(const Tensor<T> &other) const; // matrix multiplication (2D tensors)
  Tensor<T> transpose(int dim1, int dim2) const; // transpose two dimensions
  Tensor<T> sum(int dim) const;                  // sum over a dimension
  Tensor<T> mean(int dim) const;                 // mean over a dimension
  Tensor<T> max(int dim) const;                  // max over a dimension
  Tensor<T> min(int dim) const;                  // min over a dimension
  Tensor<T> argmax(int dim) const;               // index of max value along dim
  Tensor<T> argmin(int dim) const;               // index of min value along dim

  // Element-Wise comparison
  Tensor<bool> equal(const Tensor<T> &other) const;
  Tensor<bool> greater(const Tensor<T> &other) const;
  Tensor<bool> less(const Tensor<T> &other) const;
  Tensor<bool> greater_equal(const Tensor<T> &other) const;
  Tensor<bool> less_equal(const Tensor<T> &other) const;

  static Tensor<T> sum_over_broadcasted_axes(const Tensor<T> &input,
                                             const Tensor<T> &gradient);
  Tensor<T> broadcast_to(const std::vector<int> &new_shape) const;
  Tensor<T> slice(const std::vector<int> &start_indices,
                  const std::vector<int> &sizes) const;
  Tensor<T> gather(int dim, const Tensor<int> &indices) const;

  Tensor<T> reshape(const std::vector<int> &new_shape);
  void computeStrides() {
    strides.resize(shape.size());
    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= shape[i];
    }
  }

  /**
   * This is the top level function which should only be called on loss/output
   * tensor
   */

  // Operator definitons for syntactic sugar
  Tensor<T> operator*(const Tensor<T> &other) const {
    return multiply(other);
  } // ADD SMART MULTIPLICATION (AUTO MATRIX MUL FOR 2D)
  Tensor<T> operator*(const T &scalar) const { return multiply(scalar); }

  Tensor<T> operator+(const Tensor<T> &other) const { return add(other); }
  Tensor<T> operator+(const T &scalar) const { return add(scalar); }

  Tensor<T> operator-(const Tensor<T> &other) const { return subtract(other); }
  Tensor<T> operator-(const T &scalar) const { return subtract(scalar); }

  Tensor<T> operator/(const Tensor<T> &other) const { return divide(other); }
  Tensor<T> operator/(const T &scalar) const { return divide(scalar); }

  // Batch Multiplication

  static bool canBatchMultiply(const Tensor<T> &tensorA,
                               const Tensor<T> &tensorB);
  static std::vector<int> infer_broadcast_shape(const Tensor<T> &tensorA,
                                                const Tensor<T> &tensorB);
  Tensor<T> batchMultiply(const Tensor<T> &other) const;

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

  Tensor<T> sign() const;

  void print() const;

protected:
  void _backwards() {
    if (backwardsFunction)
      backwardsFunction();
  }
};

#include "Tensor.tpp"

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
_backwards
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
