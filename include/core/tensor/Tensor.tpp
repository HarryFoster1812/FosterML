#pragma once
#include "TensorPtr.hpp"
#include <algorithm> // for std::maxAdd commentMore actions
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <iostream>
#include <locale>
#include <memory>
#include <numeric> // for std::iota
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "ops/ops.hpp"

// CONSTRUCTOR FUNCTIONS

namespace FosterML {

// helper debug function to cout vectors
template <typename S>
std::ostream& operator<<(std::ostream& os, const std::vector<S>& vector) {

    // Printing all the elements using <<
    for (auto i : vector)
        os << i << " ";
    return os;
}

// GET LOGIC

template <typename T>
T& Tensor<T>::operator()(const std::initializer_list<int>& index_values) {
    // Case: scalar tensor (0-D)
    if (shape.size() == 0) {
        return (*data)[0];
    }

    // Check matching rank
    if (index_values.size() != shape.size()) {
        throw std::runtime_error(
            "Incorrect number of indices for tensor access");
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
T& Tensor<T>::operator()(const std::vector<int>& index_values) {
    // Case: scalar tensor (0-D)
    if (shape.size() == 0) {
        return (*data)[0];
    }

    // Check matching rank
    if (index_values.size() != shape.size()) {
        throw std::runtime_error(
            "Incorrect number of indices for tensor access");
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
const T& Tensor<T>::operator()(const std::vector<int>& index_values) const {
    if (shape.size() == 0) {
        return (*data)[0];
    }

    if (index_values.size() != shape.size()) {
        throw std::runtime_error(
            "Incorrect number of indices for tensor access");
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

template <typename T> TensorPtr<T> Tensor<T>::add(const T& scalar) const {
    TensorPtr<T> scalarTensor =
        TensorPtr<T>::create({}, false); // shape = {} means 0-d scalar tensor
    scalarTensor->setData({scalar});

    // Then reuse the tensor add
    return this->add(scalarTensor);
}

template <typename T>
TensorPtr<T> Tensor<T>::add(const TensorPtr<T>& other) const {
    auto node = OpNode<T>::template create<AddOp<T>>(
        TensorPtr<T>(this->shared_from_this()), other);
    node->forward();
    return node->getOutput();
}

template <typename T> TensorPtr<T> Tensor<T>::subtract(const T& scalar) const {
    TensorPtr<T> scalarTensor =
        TensorPtr<T>::create({}, false); // shape = {} means 0-d scalar tensor
    scalarTensor->setData({scalar});

    // Then reuse the tensor add
    return this->subtract(scalarTensor);
}

template <typename T>
TensorPtr<T> Tensor<T>::subtract(const TensorPtr<T>& other) const {
    auto node = OpNode<T>::template create<SubOp<T>>(
        TensorPtr<T>(this->shared_from_this()), other);
    node->forward();
    return node->getOutput();
}

template <typename T>
TensorPtr<T> Tensor<T>::multiply(const TensorPtr<T>& other) const {

    auto node = OpNode<T>::template create<MultiplyOp<T>>(
        TensorPtr<T>(this->shared_from_this()), other);
    node->forward();
    return node->getOutput();
}

template <typename T> TensorPtr<T> Tensor<T>::multiply(const T& scalar) const {
    TensorPtr<T> scalarTensor =
        TensorPtr<T>::create({}, false); // shape = {} means 0-d scalar tensor
    scalarTensor->setData({scalar});

    // Then reuse the tensor add
    return this->multiply(scalarTensor);
}

template <typename T>
TensorPtr<T> Tensor<T>::divide(const TensorPtr<T>& other) const {
    auto node = OpNode<T>::template create<DivOp<T>>(
        TensorPtr<T>(this->shared_from_this()), other);
    node->forward();
    return node->getOutput();
}

template <typename T> TensorPtr<T> Tensor<T>::divide(const T& scalar) const {
    TensorPtr<T> scalarTensor =
        TensorPtr<T>::create({}, false); // shape = {} means 0-d scalar tensor
    scalarTensor->setData({scalar});

    // Then reuse the tensor add
    return this->divide(scalarTensor);
}

// Element Wise operations

template <typename T>
TensorPtr<T> Tensor<T>::negate() const // element-wise negation (-x)
{
    auto node = OpNode<T>::template create<NegateOp<T>>(
        TensorPtr<T>(this->shared_from_this()));
    node->forward();
    return node->getOutput();
}

template <typename T>
TensorPtr<T> Tensor<T>::abs() const // element-wise negation (-x)
{
    auto node = OpNode<T>::template create<AbsOp<T>>(
        TensorPtr<T>(this->shared_from_this()));
    node->forward();
    return node->getOutput();
}

template <typename T>
TensorPtr<T> Tensor<T>::sqrt() const // element-wise square root
{
    auto node = OpNode<T>::template create<SqrtOp<T>>(
        TensorPtr<T>(this->shared_from_this()));
    node->forward();
    return node->getOutput();
}

template <typename T>
TensorPtr<T> Tensor<T>::exp() const // element-// element-wise exponentiation
{
    auto node = OpNode<T>::template create<ExpOp<T>>(
        TensorPtr<T>(this->shared_from_this()));
    node->forward();
    return node->getOutput();
}

template <typename T>
TensorPtr<T> Tensor<T>::log() const // element-// element-wise natural logarithm
{
    auto node = OpNode<T>::template create<LogOp<T>>(
        TensorPtr<T>(this->shared_from_this()));
    node->forward();
    return node->getOutput();
}

template <typename T>
TensorPtr<T> Tensor<T>::sigmoid() const // element-// element-wise sigmoid
{
    auto node = OpNode<T>::template create<SigmoidOp<T>>(
        TensorPtr<T>(this->shared_from_this()));
    node->forward();
    return node->getOutput();
}

template <typename T>
TensorPtr<T>
Tensor<T>::tanh() const // element-// element-wise hyperbolic tangent
{
    auto node = OpNode<T>::template create<TanhOp<T>>(
        TensorPtr<T>(this->shared_from_this()));
    node->forward();
    return node->getOutput();
}

// this @ other
template <typename T>
TensorPtr<T> Tensor<T>::matrixmul(const TensorPtr<T>& other) const {
    auto node = OpNode<T>::template create<MatMulOp<T>>(
        TensorPtr<T>(this->shared_from_this()), other);
    node->forward();
    return node->getOutput();
}

// Util functions

template <typename T> TensorPtr<T> Tensor<T>::sign() const {
    TensorPtr<T> result = TensorPtr<T>::create(shape, false);

    // Calculate the result
    std::vector<T> result_Data = result->getData();

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

template <typename T>
TensorPtr<T> Tensor<T>::transpose(int dim1, int dim2) const {
    // Handle negative axes
    int rank = shape.size();
    dim1 = (dim1 < 0) ? dim1 + rank : dim1;
    dim2 = (dim2 < 0) ? dim2 + rank : dim2;

    if (dim1 < 0 || dim1 >= rank || dim2 < 0 || dim2 >= rank) {
        throw std::invalid_argument("Invalid dimensions for transpose");
    }

    std::vector<int> new_shape = shape;
    std::vector<int> new_strides = strides;

    std::swap(new_shape[dim1], new_shape[dim2]);
    std::swap(new_strides[dim1], new_strides[dim2]);

    TensorPtr<T> result =
        TensorPtr<T>::create(new_shape, false); // false: no grad by default
    result->shareData(this->data);              // reuses the same data buffer
    result->setStrides(new_strides);            // only change how it's indexed
    return result;
}

template <typename T>
std::vector<int> Tensor<T>::infer_broadcast_shape(const TensorPtr<T>& tensorA,
                                                  const TensorPtr<T>& tensorB) {
    std::vector<int> shapeA = tensorA->getShape();
    std::vector<int> shapeB = tensorB->getShape();
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

template <typename T>
std::vector<int>
Tensor<T>::infer_broadcast_shape(const std::vector<int>& shapeA,
                                 const std::vector<int>& shapeB) {
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
TensorPtr<T> Tensor<T>::broadcast_to(const std::vector<int>& new_shape,
                                     bool matrix_mul) const {
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
    TensorPtr<T> result = TensorPtr<T>::create(new_shape, false);
    result->shareData(data);
    result->setStrides(new_strides);

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
void Tensor<T>::printRecursive(std::vector<int>& indices, int dim,
                               int depth) const {
    if (dim == shape.size() - 1) {
        // Base case: print the innermost dimension
        std::cout << "[";
        for (int i = 0; i < shape[dim]; ++i) {
            indices[dim] = i;
            std::cout << std::setw(8) << std::setprecision(17)
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
TensorPtr<T> Tensor<T>::sum(const std::vector<int>& axis, bool keepdims) const {
    // if axis is empty then sum over every axis
    std::vector<int> sum_axis = axis;
    if (sum_axis.empty()) {
        sum_axis.resize(shape.size());
        std::iota(sum_axis.begin(), sum_axis.end(), 0); // {0,1,2,...}
    }

    // handle case where axis is out of bounds
    for (auto& axis : sum_axis) {
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

    TensorPtr<T> result = TensorPtr<T>::create(result_shape, requires_gradient);

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

        // std::cout << "out_index after calulation for i of: " << i <<
        // std::endl; std::cout << out_index << std::endl;

        (*result)(out_index) += (*this)(unraveled_index);
    }

    return result;
}

template <typename T>
void Tensor<T>::unravel_index(int flat_index,
                              std::vector<int>& indices_out) const {
    for (int i = 0; i < shape.size(); ++i) {
        indices_out[i] = flat_index / strides[i];
        flat_index %= strides[i];
    }
}

// NOTE: This function should only be called when the two tensors have a
// different shape and the grdient is larger than the target
template <typename T>
TensorPtr<T>
Tensor<T>::sum_over_broadcasted_axes(const TensorPtr<T>& gradient,
                                     const std::vector<int>& target_shape) {
    // std::cout << "IN sum_over_broadcasted_axes" << std::endl;
    const auto grad_shape = gradient->getShape();

    int gradient_dims = gradient->getShape().size();
    int target_dims = target_shape.size();
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

    if (axis_to_sum.empty()) {
        return gradient;
    }

    // 3. Perform reduction (sum over axes_to_sum)
    return gradient->sum(axis_to_sum, false);

    // find the differing axis
    // create a vector
    // return gradient.sum(axis_to_reduce, false);
}

template <typename T> T Tensor<T>::det() const {
    size_t n = shape[0];
    if (shape.size() != 2 || shape[0] != shape[1])
        throw std::invalid_argument("input to det must be a square matrix");

    TensorPtr<T> A = clone();
    T det = 1;
    int swaps = 0;

    for (size_t i = 0; i < n; ++i) {
        // Pivoting
        size_t pivot = i;
        for (size_t row = i + 1; row < n; ++row)
            if (std::abs((*A)(row, i)) > std::abs((*A)(pivot, i)))
                pivot = row;

        if ((*A)(pivot, i) == 0)
            return 0; // Singular

        if (pivot != i) {
            A->swap_rows(i, pivot);
            swaps++;
        }

        det *= (*A)(i, i);

        for (size_t j = i + 1; j < n; ++j) {
            T factor = (*A)(j, i) / (*A)(i, i);
            for (size_t k = i; k < n; ++k)
                (*A)(j, k) -= factor * (*A)(i, k);
        }
    }

    return (swaps % 2 == 0) ? det : -det;
}

template <typename T> TensorPtr<T> Tensor<T>::pinverse() const {
    if (shape.size() != 2) {
        throw std::invalid_argument(
            "Inverse is only defined for matrix (2d tensors)");
    }

    // TensorPtr<T> aTa = matrixmul(transpose(-2, -1));
    // std::vector<T> eigen_values = Tensor<T>::eigenValues(aTa);
    // SVD returns:
    // M=U \sigma V
    //
    // Inverse is:
    // M^+=V W^{−1} U^T
    // Where W^{-1} is the reciprocol of the diagonal \sigma values but has the
    // same shape as A
}

template <typename T>
TensorPtr<T> Tensor<T>::cat(TensorPtr<T> dataToAdd, int dim) const {
    TensorPtr<T> result = TensorPtr<T>::create();
}
template <typename T> TensorPtr<T> Tensor<T>::cat(T dataFill, int dim) const {}

} // namespace FosterML

/*
TODO:
- IMPLEMENT MATRIX MUL AND BATCH MATRIX MUL AS WELL AS FIGURE OUT THE PARTAIL
DIFF FOR THEM
- ADD MORE HELPER FUNCTIONS
- FIGURE OUT HOW I CAN OPTIMISE THIS WITHOUT LOSING THE WILL TO LIVE
*/
