#pragma once
#include <algorithm> // for std::max
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <iostream>
#include <locale>
#include <numeric> // for std::iota
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

  int flat_index = 0;
  for (int i = 0; i < shape.size(); ++i) {
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

  // I HAVE GIVEN UP MAKING GOOD READABLE CODE
  Tensor<T> A_broadcasted = (A.getShape() == broadcast_shape)
                                ? Tensor<T>(A)
                                : A.broadcast_to(broadcast_shape);
  Tensor<T> B_broadcasted = (B.getShape() == broadcast_shape)
                                ? Tensor<T>(B)
                                : B.broadcast_to(broadcast_shape);

  // Create result tensor
  Tensor<T> result(broadcast_shape, A.requiresGrad() || B.requiresGrad());

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
      // std::cout << "IN THE BACKWARDS_OPERATION" << std::endl;
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
          Tensor<T> grad_A = Tensor<T>::sum_over_broadcasted_axes(
              result.getGrad(), other_ptr->getShape());
          other_ptr->addGrad(grad_A);
        } else {
          other_ptr->addGrad(result.getGrad());
        }
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

template <typename T>
Tensor<T> Tensor<T>::subtract(const Tensor<T> &other) const {

  auto add_op = [](const T &a, const T &b) { return a - b; };
  std::function<void(const Tensor<T> &)> backward_op;
  try {
    auto this_shared = this->getSharedPtr();
    auto other_shared = other.getSharedPtr();
    backward_op = [this_ptr = this_shared,
                   other_ptr = other_shared](const Tensor<T> &result) {
      // std::cout << "IN THE BACKWARDS_OPERATION" << std::endl;
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
          Tensor<T> grad_A = Tensor<T>::sum_over_broadcasted_axes(
              result.getGrad(), other_ptr->getShape());
          other_ptr->addGrad(grad_A.negate());
        } else {
          other_ptr->addGrad(result.getGrad().negate());
        }
      }
    };
  } catch (const std::bad_weak_ptr &e) {
    std::cerr << "BACKWARDS_OPERATION BAD WEAK PTR" << std::endl;
    throw;
  }

  return element_op(*this, other, add_op, backward_op);
}

template <typename T> Tensor<T> Tensor<T>::subtract(const T &scalar) const {
  Tensor<T> scalarTensor({}, false); // shape = {} means 0-d scalar tensor
  scalarTensor.setData({scalar});

  // Then reuse the tensor add
  return this->subtract(scalarTensor);
}


template <typename T>
Tensor<T> Tensor<T>::multiply(const Tensor<T> &other) const {

  auto add_op = [](const T &a, const T &b) { return a * b; };
  std::function<void(const Tensor<T> &)> backward_op;
    try {
        auto this_shared = this->getSharedPtr();
        auto other_shared = other.getSharedPtr();
        backward_op = [this_ptr = this_shared,
            other_ptr = other_shared](const Tensor<T> &result) {
                // std::cout << "IN THE BACKWARDS_OPERATION" << std::endl;
                const Tensor<T> &grad = result.getGrad();
                if (this_ptr->requiresGrad()) {
                    Tensor<T> grad_A = grad.multiply(*other_ptr);
                    if (this_ptr->getShape() != grad_A.getShape()) {
                        grad_A = Tensor<T>::sum_over_broadcasted_axes(
                            grad_A, this_ptr->getShape());
                        this_ptr->addGrad(grad_A);
                    } else {
                        this_ptr->addGrad(grad_A);
                    }
                }

                if (other_ptr->requiresGrad()) {
                    Tensor<T> grad_B = grad.multiply(*this_ptr);
                    if (other_ptr->getShape() != grad_B.getShape()) {
                        grad_B = Tensor<T>::sum_over_broadcasted_axes(
                            grad_B, other_ptr->getShape());
                        other_ptr->addGrad(grad_B);
                    } else {
                        other_ptr->addGrad(grad_B);
                    }
                }
            };
    } catch (const std::bad_weak_ptr &e) {
        std::cerr << "BACKWARDS_OPERATION BAD WEAK PTR" << std::endl;
        throw;
    }

    return element_op(*this, other, add_op, backward_op);
}

template <typename T> Tensor<T> Tensor<T>::multiply(const T &scalar) const {
  Tensor<T> scalarTensor({}, false); // shape = {} means 0-d scalar tensor
  scalarTensor.setData({scalar});

  // Then reuse the tensor add
  return this->multiply(scalarTensor);
}



template <typename T>
Tensor<T> Tensor<T>::divide(const Tensor<T> &other) const {

  auto add_op = [](const T &a, const T &b) { return a / b; };
  std::function<void(const Tensor<T> &)> backward_op;
    try {
        auto this_shared = this->getSharedPtr();
        auto other_shared = other.getSharedPtr();
        backward_op = [this_ptr = this_shared,
            other_ptr = other_shared](const Tensor<T> &result) {
                // for division C = A/B
                // \partial C w.r.t A = 1/B 
                // std::cout << "IN THE BACKWARDS_OPERATION" << std::endl;
                const Tensor<T> &grad = result.getGrad();
                if (this_ptr->requiresGrad()) {
                    Tensor<T> grad_A = grad.divide(*other_ptr);

                    if (this_ptr->getShape() != grad_A.getShape()) {
                        grad_A = Tensor<T>::sum_over_broadcasted_axes(
                            grad_A, this_ptr->getShape());
                        
                        this_ptr->addGrad(grad_A);
                    
                    } else {
                        this_ptr->addGrad(grad_A);
                    }
                }

                // \partial C w.r.t B = -A/B^2
                if (other_ptr->requiresGrad()) {
                    Tensor<T> grad_B = grad.multiply(*this_ptr).negate(); // grad * -A
                    Tensor<T> B_squred = (*other_ptr).multiply(*other_ptr); // grad * -A
                    grad_B = grad_B.divide(B_squred); // grad *(-A/b^2)
                    if (other_ptr->getShape() != grad_B.getShape()) {
                        grad_B = Tensor<T>::sum_over_broadcasted_axes(
                            grad_B, other_ptr->getShape());
                    
                        other_ptr->addGrad(grad_B);
                    
                    } else {
                        other_ptr->addGrad(grad_B);
                    }
                }
            };
    } catch (const std::bad_weak_ptr &e) {
        std::cerr << "BACKWARDS_OPERATION BAD WEAK PTR" << std::endl;
        throw;
    }

    return element_op(*this, other, add_op, backward_op);
}

template <typename T> Tensor<T> Tensor<T>::divide(const T &scalar) const {
  Tensor<T> scalarTensor({}, false); // shape = {} means 0-d scalar tensor
  scalarTensor.setData({scalar});

  // Then reuse the tensor add
  return this->divide(scalarTensor);
}

// Element Wise operations

template <typename T, typename Op >
Tensor<T> elementwise_op(const Tensor<T> &A, Op op) {
  // Broadcast shapes

  // Create result tensor
  Tensor<T> result(A.getShape(), A.requiresGrad());

  // Calculate the result

  std::vector<int> index(result.getShape().size(), 0);
  do {
    result(index) = op(A(index));
  } while (Tensor<T>::incrementIndex(index, result.getShape()));

  return result;
}

template <typename T>
Tensor<T> Tensor<T>::negate() const // element-wise negation (-x)
{
  auto op = [](const T &a) { return -a; };

  return elementwise_op(*this, op);
}
// Util functions

template <typename T>
std::vector<int> Tensor<T>::infer_broadcast_shape(const Tensor<T> &tensorA,
                                                  const Tensor<T> &tensorB) {
  std::vector<int> shapeA = tensorA.getShape();
  std::vector<int> shapeB = tensorB.getShape();
  int lenA = shapeA.size();
  int lenB = shapeB.size();
  int result_len = std::max(lenA, lenB);

  // Prepare padded shapes with leading 1s
  std::vector<int> paddedA(result_len, 1);
  std::vector<int> paddedB(result_len, 1);

  // Copy original shapes into padded vectors, right-aligned
  for (int i = 0; i < lenA; ++i) {
    paddedA[result_len - lenA + i] = shapeA[i];
  }
  for (int i = 0; i < lenB; ++i) {
    paddedB[result_len - lenB + i] = shapeB[i];
  }

  // Compute broadcasted shape
  std::vector<int> result(result_len);
  for (int i = 0; i < result_len; ++i) {
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
  int original_len = shape.size();
  int new_len = new_shape.size();

  if (new_len < original_len) {
    throw std::runtime_error(
        "Cannot broadcast to a smaller number of dimensions");
  }

  // Pad the original shape and strides with leading 1s and dummy strides
  std::vector<int> padded_shape(new_len, 1);
  std::vector<int> padded_strides(new_len,
                                  0); // Default to 0 for broadcasted dims

  for (int i = 0; i < original_len; ++i) {
    padded_shape[new_len - original_len + i] = shape[i];
    padded_strides[new_len - original_len + i] = strides[i];
  }

  // Compute new strides according to broadcasting rules
  std::vector<int> new_strides(new_len);
  for (int i = 0; i < new_len; ++i) {
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
  for (int i = 0; i < shape.size(); ++i) {
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

template <typename T>
Tensor<T> Tensor<T>::sum(const std::vector<int> &axis, bool keepdims) const {
  // if axis is empty then sum over every axis
  std::vector<int> sum_axis = axis;
  if (sum_axis.empty()) {
    sum_axis.resize(shape.size());
    std::iota(sum_axis.begin(), sum_axis.end(), 0); // {0,1,2,...}
  }

  // handle case where axis is out of bounds
  for (auto &axis : sum_axis) {
    if (axis >= shape.size())
      throw std::runtime_error("Axis out of range");
  }

  std::sort(sum_axis.begin(), sum_axis.end());

  std::vector<int> result_shape;
  for (int i = 0; i < shape.size(); ++i) {
    // if i is not in the vector given std::find will return vector.end()
    if (std::find(sum_axis.begin(), sum_axis.end(), i) != sum_axis.end()) {
      if (keepdims)
        result_shape.push_back(1);
      // else skip this dimension (reduced)
    } else {
      result_shape.push_back(shape[i]);
    }
  }

  Tensor<T> result(result_shape, requires_gradient);

  std::vector<bool> is_reduced_axis(shape.size(), false);
  for (auto a : sum_axis)
    is_reduced_axis[a] = true;

  std::vector<int> out_index(
      result_shape.size()); // allocate once before the loop
  std::vector<int> unraveled_index(
      shape.size()); // allocate once before the loop

  // std::cout << "OUT INDEX" << std::endl;
  // std::cout << out_index << std::endl;
  // std::cout << "unraveled_index" << std::endl;
  // std::cout << unraveled_index << std::endl;
  // build the output tensor
  for (int i = 0; i < (*data).size(); ++i) {

    // calculate the multi-dimensional index for the flat index
    unravel_index(i, unraveled_index);

    // std::cout << "unraveled_index after calulation for i of: " << i <<
    // std::endl; std::cout << unraveled_index << std::endl;

    // calcualte the corresponding broadcast index from the unraveled_index
    size_t out_dim = 0;
    for (size_t dim = 0; dim < shape.size(); ++dim) {
      if (is_reduced_axis[dim]) {
        if (keepdims) {
          out_index[out_dim++] = 0;
        }
      } else {
        out_index[out_dim++] = unraveled_index[dim];
      }
    }

    // std::cout << "out_index after calulation for i of: " << i << std::endl;
    // std::cout << out_index << std::endl;

    result(out_index) += (*this)(unraveled_index);
  }

  return result;
}

template <typename T>
void Tensor<T>::unravel_index(int flat_index,
                              std::vector<int> &indices_out) const {
  for (int i = 0; i < shape.size(); ++i) {
    indices_out[i] = flat_index / strides[i];
    flat_index %= strides[i];
  }
}

// NOTE: This function should only be called when the two tensors have a
// different shape and the grdient is larger than the target
template <typename T>
Tensor<T> Tensor<T>::sum_over_broadcasted_axes(const Tensor<T> &gradient,
                                               const Tensor<T> &target) {
  // std::cout << "IN sum_over_broadcasted_axes" << std::endl;
  const auto grad_shape = gradient.getShape();
  const auto target_shape = target.getShape();

  int gradient_dims = gradient.getShape().size();
  int target_dims = target.getShape().size();
  int diff_dims = gradient_dims - target_dims;
  std::vector<int> axis_to_sum;

  // 1. Add leading axes that were added during broadcast
  for (int i = 0; i < diff_dims; ++i) {
    axis_to_sum.push_back(i);
  }

  // 2. Check for axes that were broadcast within aligned dimensions
  for (int i = 0; i < target_shape.size(); ++i) {
    int grad_axis = i + diff_dims;
    if (target_shape[i] == 1 && grad_shape[grad_axis] > 1) {
      axis_to_sum.push_back(grad_axis);
    }
  }

  // 3. Perform reduction (sum over axes_to_sum)
  return gradient.sum(axis_to_sum, false);

  // find the differing axis
  // create a vector
  // return gradient.sum(axis_to_reduce, false);
}

/*
TODO:
- CREATE A SUM OVER BROADCAST AXIS SO THAT BACKWARDS PASS WORKS FOR BROADCASTED
TENSORS
- USING THE SAME FUNCTIONS IMPLEMENT OTHER (EMELENT-WISE) OPERATIONS (SUB, DIV,
MUL)
- IMPLEMENT MATRIX MUL AND BATCH MATRIX MUL AS WELL AS FIGURE OUT THE PARTAIL
DIFF FOR THEM
- ADD MORE HELPER FUNCTIONS
- FIGURE OUT HOW I CAN OPTIMISE THIS WITHOUT LOSING THE WILL TO LIVE
*/
