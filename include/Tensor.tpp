#pragma once
#include "Tensor.hpp"
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

template <typename T, typename Op>
TensorPtr<T> element_op(const TensorPtr<T>& A, const TensorPtr<T>& B, Op op,
                        std::function<void(const Tensor<T>&)> backward_op) {
    // Broadcast shapes
    std::vector<int> broadcast_shape = Tensor<T>::infer_broadcast_shape(A, B);

    // I HAVE GIVEN UP MAKING GOOD READABLE CODE
    TensorPtr<T> A_broadcasted = (A->getShape() == broadcast_shape)
                                     ? TensorPtr<T>(A)
                                     : A->broadcast_to(broadcast_shape);
    TensorPtr<T> B_broadcasted = (B->getShape() == broadcast_shape)
                                     ? TensorPtr<T>(B)
                                     : B->broadcast_to(broadcast_shape);

    // Create result tensor
    TensorPtr<T> result = TensorPtr<T>::create(
        broadcast_shape, A->requiresGrad() || B->requiresGrad());

    // Calculate the result
    std::vector<int> index(broadcast_shape.size(), 0);
    do {
        (*result)(index) = op((*A_broadcasted)(index), (*B_broadcasted)(index));
    } while (Tensor<T>::incrementIndex(index, broadcast_shape));

    // Set backward function

    if (result->requiresGrad()) {
        result->setBackwardFunction(backward_op);
        try {
            auto A_shared = A.get_shared_ptr();
            auto B_shared = B.get_shared_ptr();
            if (A->requiresGrad())
                result->addParent(A_shared);
            if (B->requiresGrad())
                result->addParent(B_shared);
        } catch (const std::bad_weak_ptr& e) {
            std::cerr << "bad_weak_ptr for in element_op" << std::endl;
            throw;
        }
    }

    return result;
}

template <typename T>
TensorPtr<T> Tensor<T>::add(const TensorPtr<T>& other) const {

    auto add_op = [](const T& a, const T& b) { return a + b; };
    std::function<void(const Tensor<T>&)> backward_op;
    try {
        auto this_shared = this->shared_from_this();
        auto other_shared = other.get_shared_ptr();
        backward_op = [this_ptr = this_shared,
                       other_ptr = other_shared](const Tensor<T>& result) {
            // std::cout << "IN THE BACKWARDS_OPERATION" << std::endl;
            if (this_ptr->requiresGrad()) {
                if (this_ptr->getShape() != result.getShape()) {
                    TensorPtr<T> grad_A = Tensor<T>::sum_over_broadcasted_axes(
                        result.getGrad(), this_ptr->getShape());
                    this_ptr->addGrad(grad_A);
                } else {
                    this_ptr->addGrad(result.getGrad());
                }
            }

            if (other_ptr->requiresGrad()) {
                if (other_ptr->getShape() != result.getShape()) {
                    TensorPtr<T> grad_A = Tensor<T>::sum_over_broadcasted_axes(
                        result.getGrad(), other_ptr->getShape());
                    other_ptr->addGrad(grad_A);
                } else {
                    other_ptr->addGrad(result.getGrad());
                }
            }
        };
    } catch (const std::bad_weak_ptr& e) {
        std::cerr << "BACKWARDS_OPERATION BAD WEAK PTR" << std::endl;
        throw;
    }

    return element_op(TensorPtr<T>(this->shared_from_this()), other, add_op,
                      backward_op);
}

template <typename T> TensorPtr<T> Tensor<T>::add(const T& scalar) const {
    TensorPtr<T> scalarTensor =
        TensorPtr<T>::create({}, false); // shape = {} means 0-d scalar tensor
    scalarTensor->setData({scalar});

    // Then reuse the tensor add
    return this->add(scalarTensor);
}

template <typename T>
TensorPtr<T> Tensor<T>::subtract(const TensorPtr<T>& other) const {

    auto add_op = [](const T& a, const T& b) { return a - b; };
    std::function<void(const Tensor<T>&)> backward_op;
    try {
        auto this_shared = this->shared_from_this();
        auto other_shared = other.get_shared_ptr();
        backward_op = [this_ptr = this_shared,
                       other_ptr = other_shared](const Tensor<T>& result) {
            // std::cout << "IN THE BACKWARDS_OPERATION" << std::endl;
            if (this_ptr->requiresGrad()) {
                if (this_ptr->getShape() != result.getShape()) {
                    TensorPtr<T> grad_A = Tensor<T>::sum_over_broadcasted_axes(
                        result.getGrad(), this_ptr->getShape());
                    this_ptr->addGrad(grad_A);
                } else {
                    this_ptr->addGrad(result.getGrad());
                }
            }

            if (other_ptr->requiresGrad()) {
                if (other_ptr->getShape() != result.getShape()) {
                    TensorPtr<T> grad_A = Tensor<T>::sum_over_broadcasted_axes(
                        result.getGrad(), other_ptr->getShape());
                    other_ptr->addGrad(grad_A->negate());
                } else {
                    other_ptr->addGrad(result.getGrad()->negate());
                }
            }
        };
    } catch (const std::bad_weak_ptr& e) {
        std::cerr << "BACKWARDS_OPERATION BAD WEAK PTR" << std::endl;
        throw;
    }

    return element_op(TensorPtr<T>(this->shared_from_this()), other, add_op,
                      backward_op);
}

template <typename T> TensorPtr<T> Tensor<T>::subtract(const T& scalar) const {
    TensorPtr<T> scalarTensor =
        TensorPtr<T>::create({}, false); // shape = {} means 0-d scalar tensor
    scalarTensor->setData({scalar});

    // Then reuse the tensor add
    return this->subtract(scalarTensor);
}

template <typename T>
TensorPtr<T> Tensor<T>::multiply(const TensorPtr<T>& other) const {

    auto add_op = [](const T& a, const T& b) { return a * b; };
    std::function<void(const Tensor<T>&)> backward_op;
    try {
        auto this_shared = this->shared_from_this();
        backward_op = [this_ptr = TensorPtr<T>(this_shared),
                       other_ptr = other](const Tensor<T>& result) {
            // std::cout << "IN THE BACKWARDS_OPERATION" << std::endl;
            const TensorPtr<T> grad = result.getGrad();
            if (this_ptr->requiresGrad()) {
                TensorPtr<T> grad_A = grad->multiply(TensorPtr<T>(other_ptr));
                if (this_ptr->getShape() != grad_A->getShape()) {
                    grad_A = Tensor<T>::sum_over_broadcasted_axes(
                        grad_A, this_ptr->getShape());
                    this_ptr->addGrad(grad_A);
                } else {
                    this_ptr->addGrad(grad_A);
                }
            }

            if (other_ptr->requiresGrad()) {
                TensorPtr<T> grad_B = grad->multiply(this_ptr);
                if (other_ptr->getShape() != grad_B->getShape()) {
                    grad_B = Tensor<T>::sum_over_broadcasted_axes(
                        grad_B, other_ptr->getShape());
                    other_ptr->addGrad(grad_B);
                } else {
                    other_ptr->addGrad(grad_B);
                }
            }
        };
    } catch (const std::bad_weak_ptr& e) {
        std::cerr << "BACKWARDS_OPERATION BAD WEAK PTR" << std::endl;
        throw;
    }

    return element_op(TensorPtr<T>(this->shared_from_this()), other, add_op,
                      backward_op);
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

    auto add_op = [](const T& a, const T& b) { return a / b; };
    std::function<void(const Tensor<T>&)> backward_op;
    try {
        auto this_shared = this->shared_from_this();
        backward_op = [this_ptr = TensorPtr<T>(this_shared),
                       other_ptr = other](const Tensor<T>& result) {
            // for division C = A/B
            // \partial C w.r.t A = 1/B
            // std::cout << "IN THE BACKWARDS_OPERATION" << std::endl;
            const TensorPtr<T>& grad = result.getGrad();
            if (this_ptr->requiresGrad()) {
                TensorPtr<T> grad_A = grad->divide(other_ptr);

                if (this_ptr->getShape() != grad_A->getShape()) {
                    grad_A = Tensor<T>::sum_over_broadcasted_axes(
                        grad_A, this_ptr->getShape());

                    this_ptr->addGrad(grad_A);

                } else {
                    this_ptr->addGrad(grad_A);
                }
            }

            // \partial C w.r.t B = -A/B^2
            if (other_ptr->requiresGrad()) {
                TensorPtr<T> grad_B =
                    grad->multiply(this_ptr)->negate(); // grad * -A
                TensorPtr<T> B_squred =
                    other_ptr->multiply(other_ptr); // grad * -A
                grad_B = grad_B->divide(B_squred);  // grad *(-A/b^2)
                if (other_ptr->getShape() != grad_B->getShape()) {
                    grad_B = Tensor<T>::sum_over_broadcasted_axes(
                        grad_B, other_ptr->getShape());

                    other_ptr->addGrad(grad_B);

                } else {
                    other_ptr->addGrad(grad_B);
                }
            }
        };
    } catch (const std::bad_weak_ptr& e) {
        std::cerr << "BACKWARDS_OPERATION BAD WEAK PTR" << std::endl;
        throw;
    }

    return element_op(TensorPtr<T>(this->shared_from_this()), other, add_op,
                      backward_op);
}

template <typename T> TensorPtr<T> Tensor<T>::divide(const T& scalar) const {
    TensorPtr<T> scalarTensor =
        TensorPtr<T>::create({}, false); // shape = {} means 0-d scalar tensor
    scalarTensor->setData({scalar});

    // Then reuse the tensor add
    return this->divide(scalarTensor);
}

// Element Wise operations

template <typename T, typename Op>
TensorPtr<T> elementwise_op(const TensorPtr<T>& A, Op op,
                            std::function<void(const Tensor<T>&)> backward_op) {
    // Broadcast shapes

    // Create result tensor
    TensorPtr<T> result =
        TensorPtr<T>::create(A->getShape(), A->requiresGrad());

    // Calculate the result

    std::vector<int> index(result->getShape().size(), 0);
    do {
        (*result)(index) = op((*A)(index));
    } while (Tensor<T>::incrementIndex(index, result->getShape()));

    if (result->requiresGrad()) {
        result->setBackwardFunction(backward_op);
        try {
            auto A_shared = A.get_shared_ptr();
            result->addParent(A_shared);
        } catch (const std::bad_weak_ptr& e) {
            std::cerr << "bad_weak_ptr for in element_op" << std::endl;
            throw;
        }
    }

    return result;
}

template <typename T>
TensorPtr<T> Tensor<T>::negate() const // element-wise negation (-x)
{
    auto op = [](const T& a) { return -a; };

    std::function<void(const Tensor<T>&)> backward_op;

    try {
        auto this_shared = this->shared_from_this();
        backward_op = [a_ptr = this_shared](const Tensor<T>& result) {
            const TensorPtr<T>& grad = result.getGrad();

            // partial z / partial x = -1
            TensorPtr<T> grad_A = grad->negate();
            a_ptr->addGrad(grad_A);
        };
    } catch (const std::bad_weak_ptr& e) {
        std::cerr << "BAD WEAK PTR IN UNARAY BACKWARDS FUNCTION" << std::endl;
    }

    return elementwise_op(TensorPtr<T>(this->shared_from_this()), op,
                          backward_op);
}

template <typename T>
TensorPtr<T> Tensor<T>::abs() const // element-wise negation (-x)
{
    auto op = [](const T& a) { return a < 0 ? -a : a; };

    std::function<void(const Tensor<T>&)> backward_op;

    try {
        auto this_shared = this->shared_from_this();
        backward_op = [a_ptr = this_shared](const Tensor<T>& result) {
            const TensorPtr<T>& grad = result.getGrad();

            // partial z / partial x = -1
            TensorPtr<T> sign_tensor = a_ptr->sign();
            a_ptr->addGrad(grad.multiply(sign_tensor));
        };
    } catch (const std::bad_weak_ptr& e) {
        std::cerr << "BAD WEAK PTR IN UNARAY BACKWARDS FUNCTION" << std::endl;
    }

    return elementwise_op(TensorPtr<T>(this->shared_from_this()), op,
                          backward_op);
}

template <typename T>
TensorPtr<T> Tensor<T>::sqrt() const // element-wise square root
{
    auto op = [](const T& a) { return std::sqrt(a); };

    std::function<void(const Tensor<T>&)> backward_op;

    try {
        auto this_shared = this->shared_from_this();
        backward_op = [a_ptr = this_shared](const Tensor<T>& result) {
            const TensorPtr<T>& grad = result.getGrad();

            // partial z / partial x = 1/(2*sqrt(x))
            TensorPtr<T> denominator = result->multiply(2);
            a_ptr->addGrad(grad->divide(denominator));
        };
    } catch (const std::bad_weak_ptr& e) {
        std::cerr << "BAD WEAK PTR IN UNARAY BACKWARDS FUNCTION" << std::endl;
    }

    return elementwise_op(TensorPtr<T>(this->shared_from_this()), op,
                          backward_op);
}

template <typename T>
TensorPtr<T> Tensor<T>::exp() const // element-// element-wise exponentiation
{
    auto op = [](const T& a) { return std::exp(a); };

    std::function<void(const Tensor<T>&)> backward_op;

    try {
        auto this_shared = this->shared_from_this();
        backward_op = [a_ptr = this_shared](const Tensor<T>& result) {
            const TensorPtr<T>& grad = result.getGrad();

            // partial z / partial x = exp(x)
            a_ptr->addGrad(
                grad->multiply(TensorPtr<T>(result.shared_from_this())));
        };
    } catch (const std::bad_weak_ptr& e) {
        std::cerr << "BAD WEAK PTR IN UNARAY BACKWARDS FUNCTION" << std::endl;
    }

    return elementwise_op(TensorPtr<T>(this->shared_from_this()), op,
                          backward_op);
}

template <typename T>
TensorPtr<T> Tensor<T>::log() const // element-// element-wise natural logarithm
{
    auto op = [](const T& a) { return std::log(a); };

    std::function<void(const Tensor<T>&)> backward_op;

    try {
        auto this_shared = this->shared_from_this();
        backward_op = [a_ptr = this_shared](const Tensor<T>& result) {
            const TensorPtr<T>& grad = result.getGrad();

            // partial z / partial x = 1/x
            a_ptr->addGrad(grad->divide(TensorPtr<T>(a_ptr)));
        };
    } catch (const std::bad_weak_ptr& e) {
        std::cerr << "BAD WEAK PTR IN UNARAY BACKWARDS FUNCTION" << std::endl;
    }

    return elementwise_op(TensorPtr<T>(this->shared_from_this()), op,
                          backward_op);
}

template <typename T>
TensorPtr<T> Tensor<T>::sigmoid() const // element-// element-wise sigmoid
{
    auto op = [](const T& a) { return (T(1)) / (T(1) + std::exp(-a)); };

    std::function<void(const Tensor<T>&)> backward_op;

    try {
        auto this_shared = this->shared_from_this();
        backward_op = [a_ptr = this_shared](const Tensor<T>& result) {
            const TensorPtr<T>& grad = result.getGrad();

            // partial z / partial x = sigmoid *(1-sigmoid)
            const TensorPtr<T> scaling_tensor =
                result.multiply(result.subtract(1));
            a_ptr->addGrad(grad->multiply(scaling_tensor));
        };
    } catch (const std::bad_weak_ptr& e) {
        std::cerr << "BAD WEAK PTR IN UNARAY BACKWARDS FUNCTION" << std::endl;
    }

    return elementwise_op(TensorPtr<T>(this->shared_from_this()), op,
                          backward_op);
}

template <typename T>
TensorPtr<T>
Tensor<T>::tanh() const // element-// element-wise hyperbolic tangent
{
    auto op = [](const T& a) { return std::tanh(a); };

    std::function<void(const Tensor<T>&)> backward_op;

    try {
        auto this_shared = this->shared_from_this();
        backward_op = [a_ptr = this_shared](const Tensor<T>& result) {
            const TensorPtr<T>& grad = result.getGrad();

            // partial z / partial x = 1-tanh^2
            const TensorPtr<T> scaling_tensor =
                result.multiply(TensorPtr<T>(result.shared_from_this()))
                    ->negate()
                    ->add(1);
            a_ptr->addGrad(grad->multiply(scaling_tensor));
        };
    } catch (const std::bad_weak_ptr& e) {
        std::cerr << "BAD WEAK PTR IN UNARAY BACKWARDS FUNCTION" << std::endl;
    }

    return elementwise_op(TensorPtr<T>(this->shared_from_this()), op,
                          backward_op);
}

std::vector<int> concat_vec(const std::vector<int>& a,
                            const std::vector<int>& b) {
    std::vector<int> result = a;
    result.insert(result.end(), b.begin(), b.end());
    return result;
}

// this @ other
template <typename T>
TensorPtr<T> Tensor<T>::matrixmul(const TensorPtr<T>& other) const {
    const auto& shapeA = this->shape;
    const auto& shapeB = other->getShape();

    int ndimA = shapeA.size();
    int ndimB = shapeB.size();

    if (ndimA < 2 || ndimB < 2) {
        throw std::invalid_argument(
            "Both tensors must be at least 2D for matrix multiplication");
    }

    int M = shapeA[ndimA - 2];
    int N = shapeA[ndimA - 1];
    int N2 = shapeB[ndimB - 2];
    int P = shapeB[ndimB - 1];

    if (N != N2) {
        throw std::invalid_argument("Inner dimensions for matmul must match: "
                                    "A(..., M, N) @ B(..., N, P)");
    }

    // 1. Extract batch dimensions
    std::vector<int> batchdimA(shapeA.begin(), shapeA.end() - 2);
    std::vector<int> batchdimB(shapeB.begin(), shapeB.end() - 2);

    std::vector<int> broadcasted_batch_shape = Tensor<T>::infer_broadcast_shape(
        TensorPtr<T>::create(batchdimA), TensorPtr<T>::create(batchdimB));

    std::vector<int> result_shape = broadcasted_batch_shape;
    result_shape.push_back(M);
    result_shape.push_back(P);

    // Broadcast inputs if needed

    TensorPtr<T> A_broadcasted =
        (batchdimA == broadcasted_batch_shape)
            ? TensorPtr<T>(this->shared_from_this())
            : this->broadcast_to(
                  concat_vec(broadcasted_batch_shape, std::vector<int>{M, N}));

    TensorPtr<T> B_broadcasted =
        (batchdimB == broadcasted_batch_shape)
            ? other
            : other->broadcast_to(
                  concat_vec(broadcasted_batch_shape, std::vector<int>{N, P}));

    // Create result tensor
    TensorPtr<T> result = TensorPtr<T>::create(
        result_shape, this->requiresGrad() || other->requiresGrad());

    // Fill the result
    int batch_size = 1;
    for (int i = 0; i < broadcasted_batch_shape.size(); ++i) {
        batch_size *= broadcasted_batch_shape[i];
    }

    std::vector<int> batch_index(broadcasted_batch_shape.size(), 0);

    do {
        for (int m = 0; m < M; ++m) {
            for (int p = 0; p < P; ++p) {
                T sum = 0;
                for (int n = 0; n < N; ++n) {
                    sum += (*A_broadcasted)(concat_vec(batch_index, {m, n})) *
                           (*B_broadcasted)(concat_vec(batch_index, {n, p}));
                }
                (*result)(concat_vec(batch_index, {m, p})) = sum;
            }
        }
    } while (incrementIndex(batch_index, broadcasted_batch_shape));

    if (result->requiresGrad()) {
        // add the backwards function
        std::shared_ptr<Tensor<T>> this_shared =
            std::const_pointer_cast<Tensor<T>>(this->shared_from_this());
        std::shared_ptr<Tensor<T>> Ab_shared = A_broadcasted.get_shared_ptr();
        std::shared_ptr<Tensor<T>> other_shared = other.get_shared_ptr();
        std::shared_ptr<Tensor<T>> Bb_shared = B_broadcasted.get_shared_ptr();

        result->setBackwardFunction(
            [a_ptr = TensorPtr<T>(this_shared),
             b_ptr = TensorPtr<T>(other_shared),
             A_bcast = TensorPtr<T>(Ab_shared),
             B_bcast = TensorPtr<T>(Bb_shared)](const Tensor<T>& result) {
                // maths... yay
                const TensorPtr<T>& grad = result.getGrad();

                // C = A@B
                if (a_ptr->requiresGrad()) {
                    TensorPtr<T> BTransposed = B_bcast->transpose(-2, -1);
                    TensorPtr<T> dA_raw = grad->matrixmul(BTransposed);
                    TensorPtr<T> dA = Tensor<T>::sum_over_broadcasted_axes(
                        dA_raw, a_ptr->getShape());
                    a_ptr->addGrad(dA);
                }

                if (b_ptr->requiresGrad()) {
                    TensorPtr<T> ATransposed = A_bcast->transpose(-2, -1);
                    TensorPtr<T> dB_raw = ATransposed->matrixmul(grad);
                    TensorPtr<T> dB = Tensor<T>::sum_over_broadcasted_axes(
                        dB_raw, b_ptr->getShape());
                    b_ptr->addGrad(dB);
                }
            });
        // add the parents
        if (this->requiresGrad())
            result->addParent(this_shared);

        if (other->requiresGrad())
            result->addParent(other_shared);
    }

    return result;
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
} // namespace FosterML

/*
TODO:
- IMPLEMENT MATRIX MUL AND BATCH MATRIX MUL AS WELL AS FIGURE OUT THE PARTAIL
DIFF FOR THEM
- ADD MORE HELPER FUNCTIONS
- FIGURE OUT HOW I CAN OPTIMISE THIS WITHOUT LOSING THE WILL TO LIVE
*/
