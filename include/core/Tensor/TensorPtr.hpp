#pragma once
#include <memory>
#include <vector>

namespace FosterML {

template <typename T> class Tensor;

template <typename T> class TensorPtr {
  private:
    std::shared_ptr<Tensor<T>> ptr;

  public:
    TensorPtr() = default;
    explicit TensorPtr(std::shared_ptr<Tensor<T>> p) : ptr(p) {}
    explicit TensorPtr(std::shared_ptr<const Tensor<T>> p_const)
        : ptr(std::const_pointer_cast<Tensor<T>>(p_const)) {}

    static TensorPtr create(const std::vector<int>& shape,
                            bool requires_grad = false) {
        return TensorPtr(std::make_shared<Tensor<T>>(shape, requires_grad));
    }

    Tensor<T>* operator->() const { return ptr.get(); }
    Tensor<T>& operator*() const { return *ptr; }
    std::shared_ptr<Tensor<T>> get_shared_ptr() const { return ptr; }
    explicit operator bool() const { return ptr != nullptr; }
    bool operator==(const TensorPtr& other) const { return ptr == other.ptr; }
    bool operator!=(const TensorPtr& other) const { return ptr != other.ptr; }
};

} // namespace FosterML
