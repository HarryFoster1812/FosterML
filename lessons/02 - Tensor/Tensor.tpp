#pragma once
#include <algorithm> // for std::max
#include <cmath>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <iostream>
#include <stdexcept>
// CONSTRUCTOR FUNCTIONS

// GET LOGIC

template <typename T>
T &Tensor<T>::operator()(const std::initializer_list<int> &index_values) const {
  // Case: scalar tensor (0-D)
  if (shape.size() == 0) {
        return &data[0];
  }

  // Check matching rank
  if (index_values.size() != shape.size()) {
    throw std::runtime_error("Incorrect number of indices for tensor access");
  }

  // Calculate flat index from multi-dimensional index
  int flat_index = 0;
  int stride = 1;

  // Assume row-major order: last dimension changes fastest
  for (int i = shape.size() - 1; i >= 0; --i) {
    int idx = *(index_values.begin() + i);
    if (idx < 0 || idx >= shape[i]) {
      throw std::runtime_error("Index out of bounds at dimension " +
                               std::to_string(i));
    }
    flat_index += idx * stride;
    stride *= shape[i];
  }

  return &data[flat_index];
}


template <typename T>
T &Tensor<T>::operator()(const std::vector<int> &index_values) const {
  // Case: scalar tensor (0-D)
  if (shape.size() == 0) {
    return &data[0];
  }

  // Check matching rank
  if (index_values.size() != shape.size()) {
    throw std::runtime_error("Incorrect number of indices for tensor access");
  }

  // Calculate flat index from multi-dimensional index
  int flat_index = 0;
  int stride = 1;

  // Assume row-major order: last dimension changes fastest
  for (int i = shape.size() - 1; i >= 0; --i) {
    int idx = *(index_values.begin() + i);
    if (idx < 0 || idx >= shape[i]) {
      throw std::runtime_error("Index out of bounds at dimension " +
                               std::to_string(i));
    }
    flat_index += idx * stride;
    stride *= shape[i];
  }

  return &data[flat_index];
}

template <typename T, typename Op, typename BackwardOp>
Tensor<T> element_op(const Tensor<T> &A, const Tensor<T> &B, Op op,
                     BackwardOp backward_op) {
  // Broadcast shapes
  std::vector<int> broadcast_shape = Tensor<T>::infer_broadcast_shape(A, B);
  Tensor<T> A_broadcasted =
      (A.getShape() == broadcast_shape) ? A : A.broadcast_to(broadcast_shape);
  Tensor<T> B_broadcasted =
      (B.getShape() == broadcast_shape) ? B : B.broadcast_to(broadcast_shape);

  // Create result tensor
  Tensor<T> result(broadcast_shape, A.requiresGrad() || B.requiresGrad());

  // Calculate the result
  std::vector<int> index(broadcast_shape.size(), 0);
  do {
    result(index) = op(A_broadcasted(index), B_broadcasted(index));
  } while (Tensor<T>::incrementIndex(index, broadcast_shape));

  result.addParent(A);
  result.addParent(B);

  // Set backward function
  if (result.requiresGrad())
    result.setBackwardsFunction(backward_op);

  return result;
}

template <typename T> Tensor<T> Tensor<T>::add(const Tensor<T> &other) const {

  auto add_op = [](const T &a, const T &b) { return a + b; };

  auto backward_op = [this_ptr = std::make_shared<Tensor<T>>(*this),
                      other_ptr = std::make_shared<Tensor<T>>(other)](
                         Tensor<T> &result) {
    if (this_ptr->requiresGrad()) {
      if (this_ptr->getShape() != result.getShape()) {
        Tensor<T> grad_A = Tensor<T>::sum_over_broadcasted_axes(
            result.getGrad(), this_ptr->getShape());
        this_ptr->addGrad(grad_A);
      } else {
        this_ptr->addGrad(result.getGrad());
      }
    }

    if (other_ptr->requiresGrad()) {
      if (other_ptr->getShape() != result.getShape()) {
        Tensor<T> grad_B =
            sum_over_broadcasted_axes(result.getGrad(), other_ptr->getShape());
        other_ptr->addGrad(grad_B);
      } else {
        other_ptr->addGrad(result.getGrad());
      }
    }
  };

  return element_op(*this, other, add_op, backward_op);
}

template <typename T> Tensor<T> Tensor<T>::add(const T &scalar) const {
  // convert the scalar into a tensor
  Tensor<T> scalarTensor({}, false); // shape = {} means 0-d scalar tensor
  scalarTensor.setData({scalar});

  // Then reuse the tensor add
  return this->add(scalarTensor);
}

template <typename T>
Tensor<T> Tensor<T>::subtract(const Tensor<T> &other) const {
  auto sub_op = [](const T &a, const T &b) { return a - b; };

  auto backward_op = [this_ptr = std::make_shared<Tensor<T>>(*this),
                      other_ptr = std::make_shared<Tensor<T>>(other)](
                         Tensor<T> &result) {
    if (this_ptr->requiresGrad()) {
      if (this_ptr->getShape() != result.getShape()) {
        Tensor<T> grad_A = Tensor<T>::sum_over_broadcasted_axes(
            result.getGrad(), this_ptr->getShape());
        this_ptr->addGrad(grad_A);
      } else {
        this_ptr->addGrad(result.getGrad());
      }
    }

    if (other_ptr->requiresGrad()) {
      if (other_ptr->getShape() != result.getShape()) {
        Tensor<T> grad_B =
            sum_over_broadcasted_axes(result.getGrad(), other_ptr->getShape())
                .negate();
        other_ptr->addGrad(grad_B);
      } else {
        other_ptr->addGrad(result.getGrad().negate());
      }
    }
  };

  return element_op(*this, other, sub_op, backward_op);
}

template <typename T> Tensor<T> Tensor<T>::subtract(const T &scalar) const {
  // convert the scalar into a tensor
  Tensor<T> scalarTensor({}, false); // shape = {} means 0-d scalar tensor
  scalarTensor.setData({scalar});

  // Then reuse the tensor add
  return this->subtract(scalarTensor);
}

template <typename T>
Tensor<T> Tensor<T>::multiply(const Tensor<T> &other) const {
  auto op = [](const T &a, const T &b) { return a * b; };

  auto backward_op =
      [a_ptr = std::make_shared<Tensor<T>>(*this),
       b_ptr = std::make_shared<Tensor<T>>(other)](const Tensor<T> &result) {
        const Tensor<T> &grad = result.getGrad();

        if (a_ptr->requiresGrad()) {
          Tensor<T> grad_A = grad * (*b_ptr);
          if (a_ptr->getShape() != result.getShape()) {
            grad_A = sum_over_broadcasted_axes(grad_A, a_ptr->getShape());
          }
          a_ptr->addGrad(grad_A);
        }

        if (b_ptr->requiresGrad()) {
          Tensor<T> grad_B = grad * (*a_ptr);
          if (b_ptr->getShape() != result.getShape()) {
            grad_B = sum_over_broadcasted_axes(grad_B, b_ptr->getShape());
          }
          b_ptr->addGrad(grad_B);
        }
      };

  return element_op(*this, other, op, backward_op);
}
template <typename T> Tensor<T> Tensor<T>::multiply(const T &scalar) const {
  // convert the scalar into a tensor
  Tensor<T> scalarTensor({}, false); // shape = {} means 0-d scalar tensor
  scalarTensor.setData({scalar});

  // Then reuse the tensor add
  return this->multiply(scalarTensor);
}

template <typename T>
Tensor<T> Tensor<T>::divide(const Tensor<T> &other) const {
  auto op = [](const T &a, const T &b) { return a / b; };

  auto backward_op =
      [a_ptr = std::make_shared<Tensor<T>>(*this),
       b_ptr = std::make_shared<Tensor<T>>(other)](const Tensor<T> &result) {
        const Tensor<T> &grad = result.getGrad();

        // partial z / partial x = 1/y
        if (a_ptr->requiresGrad()) {
          Tensor<T> grad_A = grad / (*b_ptr);
          if (a_ptr->getShape() != result.getShape()) {
            grad_A = sum_over_broadcasted_axes(grad_A, a_ptr->getShape());
          }
          a_ptr->addGrad(grad_A);
        }
        //  partial z / partial x = -x/y^2
        if (b_ptr->requiresGrad()) {
          Tensor<T> b_squared = (*b_ptr) * (*b_ptr);
          Tensor<T> grad_B = (grad * ((*a_ptr) / b_squared)).negate();

          if (b_ptr->getShape() != result.getShape()) {
            grad_B = sum_over_broadcasted_axes(grad_B, b_ptr->getShape());
          }
          b_ptr->addGrad(grad_B);
        }
      };

  return element_op(*this, other, op, backward_op);
}
template <typename T> Tensor<T> Tensor<T>::divide(const T &scalar) const {
  // convert the scalar into a tensor
  Tensor<T> scalarTensor({}, false); // shape = {} means 0-d scalar tensor
  scalarTensor.setData({scalar});

  // Then reuse the tensor add
  return this->divide(scalarTensor);
}

// Element Wise operations

template <typename T, typename Op, typename BackwardOp>
Tensor<T> elementwise_op(const Tensor<T> &A, Op op, BackwardOp backward_op) {
  // Broadcast shapes

  // Create result tensor
  Tensor<T> result(A.getShape(), A.requiresGrad());

  // Calculate the result
  std::vector<T> A_Data = A.getData();
  std::vector<T> result_Data = A.getData();

  for (int i = 0; i < A_Data.size(); ++i) {
    result_Data[i] = op(A_Data[i]);
  }
  result.addParent(A);

  // Set backward function
  if (result.requiresGrad())
    result.setBackwardsFunction(backward_op);

  return result;
}

template <typename T>
Tensor<T> Tensor<T>::negate() const // element-wise negation (-x)
{
  auto op = [](const T &a) { return -a; };

  auto backward_op =
      [a_ptr = std::make_shared<Tensor<T>>(*this)](const Tensor<T> &result) {
        const Tensor<T> &grad = result.getGrad();

        // partial z / partial x = -1
        Tensor<T> grad_A = grad.negate();
        a_ptr->addGrad(grad_A);
      };

  return element_op(*this, op, backward_op);
}

template <typename T>
Tensor<T> Tensor<T>::abs() const // element-wise absolute value
{
  auto op = [](const T &a) { return a < 0 ? -a : a; };
  auto backward_op =
      [a_ptr = std::make_shared<Tensor<T>>(*this)](const Tensor<T> &result) {
        const Tensor<T> &grad = result.getGrad();

        // partial z / partial x = -1
        Tensor<T> sign_tensor = (*a_ptr).sign();
        a_ptr->addGrad(grad * sign_tensor);
      };

  return element_op(*this, op, backward_op);
}

template <typename T>
Tensor<T> Tensor<T>::sqrt() const // element-wise square root
{
  auto op = [](const T &a) { return std::sqrt(a); };

  auto backward_op =
      [a_ptr = std::make_shared<Tensor<T>>(*this)](const Tensor<T> &result) {
        const Tensor<T> &grad = result.getGrad();
        // partial z / partial x = 1 / (2 * sqrt(x)) = 0.5 * x^(-0.5)
        // We can rewrite this as: 0.5 / sqrt(x)
        // Since result = sqrt(x), we can use: 0.5 / result
        Tensor<T> grad_A = grad / (result * T(2));
        a_ptr->addGrad(grad_A);
      };
  return elementwise_op(*this, op, backward_op);
}

template <typename T>
Tensor<T> Tensor<T>::exp() const // element-wise exponentiation
{
  auto op = [](const T &a) { return std::exp(a); };

  auto backward_op =
      [a_ptr = std::make_shared<Tensor<T>>(*this)](const Tensor<T> &result) {
        // partial z / partial x = exp(x) = z
        a_ptr->addGrad(result);
      };
  return elementwise_op(*this, op, backward_op);
}

template <typename T>
Tensor<T> Tensor<T>::log() const // element-wise natural logarithm
{
  auto op = [](const T &a) { return std::log(a); };

  auto backward_op =
      [a_ptr = std::make_shared<Tensor<T>>(*this)](const Tensor<T> &result) {
        // partial z / partial x = 1/x
        const Tensor<T> &grad = result.getGrad();
        a_ptr->addGrad(grad / (*a_ptr));
      };
  return elementwise_op(*this, op, backward_op);
}

template <typename T>
Tensor<T> Tensor<T>::sigmoid() const // element-wise sigmoid
{
  auto op = [](const T &a) { return (T(1)) / (T(1) + std::exp(-a)); };

  auto backward_op =
      [a_ptr = std::make_shared<Tensor<T>>(*this)](const Tensor<T> &result) {
        // partial z / partial x = 1/x
        const Tensor<T> grad = result.getGrad();
        const Tensor<T> sig_grad = grad * result * (T(1) - result);
        a_ptr->addGrad(sig_grad);
      };
  return elementwise_op(*this, op, backward_op);
}

template <typename T>
Tensor<T> Tensor<T>::tanh() const // element-wise hyperbolic tangent
{
  auto op = [](const T &a) { return (std::tanh(a)); };

  auto backward_op =
      [a_ptr = std::make_shared<Tensor<T>>(*this)](const Tensor<T> &result) {
        // partial z / partial x = 1-tanh^2
        const Tensor<T> &grad = result.getGrad();

        const Tensor<T> tanh_grad = grad * (T(1) - result * result);
        a_ptr->addGrad(tanh_grad);
      };
  return elementwise_op(*this, op, backward_op);
}

// Linear Algrbra Operations
template <typename T>
Tensor<T> Tensor<T>::matmul(
    const Tensor<T> &other) const // matrix multiplication (2D tensors)
{}

template <typename T>
Tensor<T> Tensor<T>::transpose(int dim1,
                               int dim2) const // transpose two dimensions
{}

template <typename T>
Tensor<T> Tensor<T>::sum(int dim) const // sum over a dimension
{}

template <typename T>
Tensor<T> Tensor<T>::mean(int dim) const // mean over a dimension
{}

template <typename T>
Tensor<T> Tensor<T>::max(int dim) const // max over a dimension
{}

template <typename T>
Tensor<T> Tensor<T>::min(int dim) const // min over a dimension
{}

template <typename T>
Tensor<T> Tensor<T>::argmax(int dim) const // index of max value along dim
{}

template <typename T>
Tensor<T> Tensor<T>::argmin(int dim) const // index of min value along dim
{}

// Element-Wise comparison

template <typename T>
Tensor<bool> Tensor<T>::equal(const Tensor<T> &other) const {}

template <typename T>
Tensor<bool> Tensor<T>::greater(const Tensor<T> &other) const {}

template <typename T>
Tensor<bool> Tensor<T>::less(const Tensor<T> &other) const {}

template <typename T>
Tensor<bool> Tensor<T>::greater_equal(const Tensor<T> &other) const {}

template <typename T>
Tensor<bool> Tensor<T>::less_equal(const Tensor<T> &other) const {}

template <typename T> Tensor<T> Tensor<T>::sign() const {
  Tensor<T> result(shape, false);

  // Calculate the result
  std::vector<T> result_Data = result.getData();

  for (int i = 0; i < data.size(); ++i) {
    if (data > 0)
      result_Data[i] = T(1);
    else if (data < 0)
      result_Data[i] = T(-1);
    else
      result_Data[i] = T(0);
  }
  return result;
}

// Util functions

template <typename T>
std::vector<int> Tensor<T>::infer_broadcast_shape(const Tensor<T> &tensorA,
                                                  const Tensor<T> &tensorB) {
  std::vector<int> shapeA = tensorA.getShape();
  std::vector<int> shapeB = tensorB.getShape();
  size_t lenA = shapeA.size();
  size_t lenB = shapeB.size();
  size_t result_len = std::max(lenA, lenB);

  // Prepare padded shapes with leading 1s
  std::vector<int> paddedA(result_len, 1);
  std::vector<int> paddedB(result_len, 1);

  // Copy original shapes into padded vectors, right-aligned
  for (size_t i = 0; i < lenA; ++i) {
    paddedA[result_len - lenA + i] = shapeA[i];
  }
  for (size_t i = 0; i < lenB; ++i) {
    paddedB[result_len - lenB + i] = shapeB[i];
  }

  // Compute broadcasted shape
  std::vector<int> result(result_len);
  for (size_t i = 0; i < result_len; ++i) {
    int dimA = paddedA[i];
    int dimB = paddedB[i];

    if (dimA == dimB || dimA == 1 || dimB == 1) {
      result[i] = std::max(dimA, dimB);
    } else {
      throw std::runtime_error("Shapes not compatible for broadcasting");
    }
  }

  return result;
}

// this will return a "view" tensor which have the same internal data but
// different strides, shape values
template <typename T>
Tensor<T> Tensor<T>::broadcast_to(const std::vector<int> &new_shape) const {}

template <typename T>
Tensor<T> Tensor<T>::slice(const std::vector<int> &start_indices,
                           const std::vector<int> &sizes) const {}

template <typename T>
Tensor<T> Tensor<T>::gather(int dim, const Tensor<int> &indices) const {}

template <typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<int> &new_shape) {
  int old_total = 1, new_total = 1;
  for (int dim : shape)
    old_total *= dim;
  for (int dim : new_shape)
    new_total *= dim;

  if (old_total != new_total)
    throw std::runtime_error("Reshape failed: incompatible shape sizes");

    Tensor<T> result(new_shape, requires_gradient);
    result.setData(data);
  return result;
}

template <typename T> void Tensor<T>::print() const {
  std::cout << "Tensor(shape=[";
  for (size_t i = 0; i < shape.size(); ++i) {
    std::cout << shape[i];
    if (i < shape.size() - 1)
      std::cout << ", ";
  }
  std::cout << "])" << std::endl;

  if (shape.empty()) {
    // 0D tensor (scalar)
    std::cout << data[0] << std::endl;
    return;
  }

  // Print the tensor data
  printRecursive(std::vector<int>(shape.size(), 0), 0, 0);
  std::cout << std::endl;
}

// Helper function for recursive printing with better formatting
template <typename T>
void Tensor<T>::printRecursive(std::vector<int> indices, int dim,
                               int depth) const {
  if (dim == shape.size() - 1) {
    // Base case: print the innermost dimension
    std::cout << "[";
    for (int i = 0; i < shape[dim]; ++i) {
      indices[dim] = i;
      std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                << (*this)(indices);
      if (i < shape[dim] - 1)
        std::cout << ", ";
    }
    std::cout << "]";
  } else {
    // Recursive case: handle outer dimensions
    std::cout << "[";
    bool first = true;
    for (int i = 0; i < shape[dim]; ++i) {
      indices[dim] = i;
      if (!first) {
        std::cout << "," << std::endl;
        // Add tabs based on depth
        for (int j = 0; j <= depth; ++j)
          std::cout << "\t";
      }
      first = false;
      printRecursive(indices, dim + 1, depth + 1);
    }
    std::cout << "]";
  }
}
