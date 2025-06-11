#pragma once
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>
namespace FosterML {

template <typename T> class AutoDiffEngine;
template <typename T> class Tensor;

// Wrapper class for shared_ptr<Tensor<T>>
template <typename T> class TensorPtr {
  private:
    std::shared_ptr<Tensor<T>> ptr;

  public:
    // Constructors
    TensorPtr() = default;
    explicit TensorPtr(std::shared_ptr<Tensor<T>> p) : ptr(p) {}

    explicit TensorPtr(std::shared_ptr<const Tensor<T>> p_const)
        : ptr(std::const_pointer_cast<Tensor<T>>(p_const)) {}

    // Create new tensor
    static TensorPtr create(const std::vector<int>& shape,
                            bool requires_grad = false) {
        return TensorPtr(std::make_shared<Tensor<T>>(shape, requires_grad));
    }

    // Access underlying pointer
    Tensor<T>* operator->() const { return ptr.get(); }
    Tensor<T>& operator*() const { return *ptr; }

    // Get shared_ptr
    std::shared_ptr<Tensor<T>> get_shared_ptr() const { return ptr; }

    // Check if valid
    explicit operator bool() const { return ptr != nullptr; }

    // Comparison
    bool operator==(const TensorPtr& other) const { return ptr == other.ptr; }
    bool operator!=(const TensorPtr& other) const { return ptr != other.ptr; }
};

template <typename T>
class Tensor : public std::enable_shared_from_this<Tensor<T>> {
    friend class AutoDiffEngine<T>;

  private:
    std::shared_ptr<std::vector<T>> data;
    std::vector<int> shape;
    std::vector<int> strides;
    TensorPtr<T> gradient;
    bool requires_gradient = false;
    std::vector<std::shared_ptr<Tensor<T>>> parents;
    std::function<void(const Tensor<T>&)> backwardsFunction;
    std::string debugName;

    void printRecursive(std::vector<int>& indices, int dim, int depth) const;

    void computeStrides() {
        strides.resize(shape.size());
        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

  public:
    // Constructor (private, use TensorPtr::create)

    Tensor(const std::vector<int> shape_, bool requires_grad = false)
        : shape(shape_), requires_gradient(requires_grad) {
        computeStrides();
        int total_size = 1;
        for (int dim : shape)
            total_size *= dim;
        data = std::make_shared<std::vector<T>>(total_size);
        if (requires_gradient)
            gradient = TensorPtr<T>::create(shape_, false);
    }

    // Delete copy/move to enforce shared_ptr usage
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) = delete;
    Tensor& operator=(Tensor&&) = delete;

    void setDebugName(std::string name) { debugName = name; }

    T& operator()(const std::initializer_list<int>& index_values);
    T& operator()(const std::vector<int>& index_values);
    const T& operator()(const std::vector<int>& index_values) const;

    const std::vector<int>& getShape() const { return shape; }
    const std::vector<T>& getData() const { return *data; }
    const TensorPtr<T>& getGrad() const {
        if (!gradient)
            throw std::runtime_error("Gradient not initialized");
        return gradient;
    }

    void addParent(std::shared_ptr<Tensor<T>> parent) {
        parents.push_back(parent);
    }

    bool requiresGrad() const { return requires_gradient; }

    void zeroGrad() {
        if (requires_gradient && gradient) {
            gradient->fill(static_cast<T>(0));
        }
    }

    // this should only be called in the backwards function and if it known
    // that the tensor requires a gradient
    void addGrad(const TensorPtr<T>& grad) const {
        if (requires_gradient) {
            gradient->addInPlace(*grad);
        }
    }

    void addInPlace(const Tensor<T>& other) {
        if (data->size() != other.data->size())
            throw std::runtime_error("Gradient size mismatch in addInPlace");
        for (size_t i = 0; i < data->size(); ++i) {
            (*data)[i] += (*other.data)[i];
        }
    }

    void setGradOnes() {
        if (requires_gradient) {
            if (!gradient) {
                gradient = TensorPtr<T>::create(shape, false);
            }
            gradient->fill(static_cast<T>(1));
        }
    }

    void fill(T value) { std::fill(data->begin(), data->end(), value); }

    void setData(const std::vector<T>& new_data) {
        if (!data)
            throw std::runtime_error("Data pointer is null in setData");
        if (new_data.size() != data->size()) {
            throw std::runtime_error("Size mismatch in setData");
        }
        *data = new_data;
    }

    void setStrides(const std::vector<int>& new_strides) {
        strides = new_strides;
    }

    void shareData(const std::shared_ptr<std::vector<T>>& shared_data) {
        data = shared_data;
    }

    void setBackwardFunction(std::function<void(const Tensor<T>&)> func) {
        backwardsFunction = func;
    }

    // Core Arithmetic
    TensorPtr<T> add(const TensorPtr<T>& other) const;
    TensorPtr<T> add(const T& scalar) const;
    TensorPtr<T> subtract(const TensorPtr<T>& other) const;
    TensorPtr<T> subtract(const T& scalar) const;
    TensorPtr<T> multiply(const TensorPtr<T>& other) const;
    TensorPtr<T> multiply(const T& scalar) const;
    TensorPtr<T> divide(const TensorPtr<T>& other) const;
    TensorPtr<T> divide(const T& scalar) const;

    // Element Wise operations
    TensorPtr<T> negate() const;
    TensorPtr<T> abs() const;
    TensorPtr<T> sqrt() const;
    TensorPtr<T> exp() const;
    TensorPtr<T> log() const;
    TensorPtr<T> sigmoid() const;
    TensorPtr<T> tanh() const;
    TensorPtr<T> sign() const;

    TensorPtr<T> matrixmul(const TensorPtr<T>& other) const;
    TensorPtr<T> sum(const std::vector<int>& axis, bool keepdims) const;
    TensorPtr<T> broadcast_to(const std::vector<int>& new_shape,
                              bool matrix_mul = false) const;
    TensorPtr<T> transpose(int dim1, int dim2) const;

    static std::vector<int> infer_broadcast_shape(const TensorPtr<T>& tensorA,
                                                  const TensorPtr<T>& tensorB);
    static std::vector<int>
    infer_broadcast_shape(const std::vector<int>& shapeA,
                          const std::vector<int>& shapeB);
    static bool incrementIndex(std::vector<int>& index,
                               const std::vector<int>& shape) {
        for (int dim = shape.size() - 1; dim >= 0; --dim) {
            index[dim]++;
            if (index[dim] < shape[dim]) {
                return true; // successful increment
            }
            index[dim] = 0; // carry to next higher dimension
        }
        return false; // overflow, end of iteration
    }
    static TensorPtr<T>
    sum_over_broadcasted_axes(const TensorPtr<T>& gradient,
                              const std::vector<int>& target_shape);

    void unravel_index(int flat_index, std::vector<int>& indices_out) const;
    void print() const;
};
} // namespace FosterML

#include "AutoDiffEngine.hpp"
#include "Tensor.tpp"
