#pragma once
#include <algorithm> // for std::max
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <iostream>
#include <locale>
#include <ostream>
#include <stdexcept>
#include <vector>
// CONSTRUCTOR FUNCTIONS

// helper debug function to cout vectors
template <typename S>
std::ostream &operator<<(std::ostream &os, const std::vector<S> &vector) {

  // Printing all the elements using <<
  for (auto i : vector)
    os << i << " ";
  return os;
}

// GET LOGIC

template <typename T>
T &Tensor<T>::operator()(const std::initializer_list<int> &index_values) {
  // Case: scalar tensor (0-D)
  if (shape.size() == 0) {
    return (*data)[0];
  }

  // Check matching rank
  if (index_values.size() != shape.size()) {
    throw std::runtime_error("Incorrect number of indices for tensor access");
  }

  // Calculate flat index from multi-dimensional index
  int flat_index = 0;

  // Assume row-major order: last dimension changes fastest
  for (int i = shape.size() - 1; i >= 0; --i) {
    int idx = *(index_values.begin() + i);
    if (idx < 0 || idx >= shape[i]) {
      throw std::runtime_error("Index out of bounds at dimension " +
                               std::to_string(i));
    }
    flat_index += idx * strides[i];
  }

  return (*data)[flat_index];
}

template <typename T>
T &Tensor<T>::operator()(const std::vector<int> &index_values) {
  // Case: scalar tensor (0-D)
  if (shape.size() == 0) {
    return (*data)[0];
  }

  // Check matching rank
  if (index_values.size() != shape.size()) {
    throw std::runtime_error("Incorrect number of indices for tensor access");
  }

  // Calculate flat index from multi-dimensional index
  int flat_index = 0;

  // Assume row-major order: last dimension changes fastest
  for (int i = shape.size() - 1; i >= 0; --i) {
    int idx = *(index_values.begin() + i);
    if (idx < 0 || idx >= shape[i]) {
      throw std::runtime_error("Index out of bounds at dimension " +
                               std::to_string(i));
    }
    flat_index += idx * strides[i];
  }
  return (*data)[flat_index];
}

template <typename T>
const T &Tensor<T>::operator()(const std::vector<int> &index_values) const {
  if (shape.size() == 0) {
    return (*data)[0];
  }

  if (index_values.size() != shape.size()) {
    throw std::runtime_error("Incorrect number of indices for tensor access");
  }

  size_t flat_index = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    int idx = index_values[i];
    if (idx < 0 || idx >= shape[i]) {
      throw std::runtime_error("Index out of bounds at dimension " +
                               std::to_string(i));
    }
    flat_index += idx * strides[i];
  }

  return (*data)[flat_index];
}

template <typename T, typename Op>
Tensor<T> element_op(const Tensor<T> &A, const Tensor<T> &B, Op op,
                     std::function<void(const Tensor<T> &)> backward_op) {
  // Broadcast shapes
  std::vector<int> broadcast_shape = Tensor<T>::infer_broadcast_shape(A, B);
  Tensor<T> A_broadcasted = (A.getShape() == broadcast_shape)
                                ? Tensor<T>(A)
                                : A.broadcast_to(broadcast_shape);
  Tensor<T> B_broadcasted = (B.getShape() == broadcast_shape)
                                ? Tensor<T>(B)
                                : B.broadcast_to(broadcast_shape);

  // Create result tensor
  Tensor<T> result(broadcast_shape, A.requiresGrad() || B.requiresGrad());

  // ADD PARENTS

  // Set backward_op, e.g., result.setBackwardOp(backward_op);

  // Calculate the result
  std::vector<int> index(broadcast_shape.size(), 0);
  do {
    result(index) = op(A_broadcasted(index), B_broadcasted(index));
  } while (Tensor<T>::incrementIndex(index, broadcast_shape));

  // Set backward function

  if (result.requiresGrad()) {
    result.setBackwardFunction(backward_op);
    try {
      auto A_shared = A.getSharedPtr();
      auto B_shared = B.getSharedPtr();
      if (A.requiresGrad())
        result.addParent(A_shared);
      if (B.requiresGrad())
        result.addParent(B_shared);
    } catch (const std::bad_weak_ptr &e) {
      std::cerr << "bad_weak_ptr for in element_op" << std::endl;
      throw;
    }
  }

  return result;
}

template <typename T> Tensor<T> Tensor<T>::add(const Tensor<T> &other) const {

  auto add_op = [](const T &a, const T &b) { return a + b; };
  std::function<void(const Tensor<T> &)> backward_op;
  try {
    auto this_shared = this->getSharedPtr();
    auto other_shared = other.getSharedPtr();
    backward_op = [this_ptr = this_shared,
                   other_ptr = other_shared](const Tensor<T> &result) {
      if (this_ptr->requiresGrad()) {
        this_ptr->addGrad(result.getGrad());
      }

      if (other_ptr->requiresGrad()) {
        other_ptr->addGrad(result.getGrad());
      }
    };
  } catch (const std::bad_weak_ptr &e) {
    std::cerr << "BACKWARDS_OPERATION BAD WEAK PTR" << std::endl;
    throw;
  }

  return element_op(*this, other, add_op, backward_op);
}

template <typename T> Tensor<T> Tensor<T>::add(const T &scalar) const {
  Tensor<T> scalarTensor({}, false); // shape = {} means 0-d scalar tensor
  scalarTensor.setData({scalar});

  // Then reuse the tensor add
  return this->add(scalarTensor);
}
// Element Wise operations

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
Tensor<T> Tensor<T>::broadcast_to(const std::vector<int> &new_shape) const {
  size_t original_len = shape.size();
  size_t new_len = new_shape.size();

  if (new_len < original_len) {
    throw std::runtime_error(
        "Cannot broadcast to a smaller number of dimensions");
  }

  // Pad the original shape and strides with leading 1s and dummy strides
  std::vector<int> padded_shape(new_len, 1);
  std::vector<int> padded_strides(new_len,
                                  0); // Default to 0 for broadcasted dims

  for (size_t i = 0; i < original_len; ++i) {
    padded_shape[new_len - original_len + i] = shape[i];
    padded_strides[new_len - original_len + i] = strides[i];
  }

  // Compute new strides according to broadcasting rules
  std::vector<int> new_strides(new_len);
  for (size_t i = 0; i < new_len; ++i) {
    if (padded_shape[i] == new_shape[i]) {
      new_strides[i] = padded_strides[i];
    } else if (padded_shape[i] == 1) {
      // Broadcasting this dimension, so stride is 0
      new_strides[i] = 0;
    } else {
      throw std::runtime_error("Shape mismatch: cannot broadcast tensor");
    }
  }

  // Create a new tensor that shares the same data pointer, but with new shape
  // and strides
  Tensor<T> result(new_shape, false);
  result.shareData(data);
  result.setStrides(new_strides);

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
    std::cout << (*data)[0] << std::endl;
    return;
  }

  std::vector<int> indices(shape.size(), 0);
  // Print the tensor data
  printRecursive(indices, 0, 0);
  std::cout << std::endl;
}

// Helper function for recursive printing with better formatting
template <typename T>
void Tensor<T>::printRecursive(std::vector<int> &indices, int dim,
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

// NOW TRY AND DO BACKWARDS PASS
