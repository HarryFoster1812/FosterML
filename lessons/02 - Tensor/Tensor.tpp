
// CONSTRUCTOR FUNCTIONS


template <typename T>
Tensor<T> Tensor<T>::add(const Tensor<T> &other) {
  // check if addition is defined
  //  broadcast shape

  // create result tensor
  Tensor<T> result(, this->requiresGrad() || other.requiresGrad());

  // claculte the result

  result.setBackwardsFunction([]() {
    if (this.requires_grad) {
      if (this.shape != result.getShape()) {
        // Broadcasting happened for A, so reduce grad_C over broadcasted axes
        Tensor<T> grad_A = sum_over_broadcasted_axes(C.getGrad(), this.getShape());

        this.addGrad(grad_A);
      } else {
        // No broadcasting, just add directly
        this.addGrad(result.getGrad());
      }
    }
    if (other.requires_grad()) {
      if (other.getShape() != result.getShape()) {
        // Broadcasting happened for B, reduce accordingly
        Tensor<T> grad_B =
            sum_over_broadcasted_axes(result.getGrad(), other.getShape());
        B.addGrad(grad_B);
      } else {
        B.addGrad(result.getGrad());
      }
    }
  });
}



template <typename T> 
Tensor<T> Tensor<T>::add(const T& scalar) const;

template <typename T>
Tensor<T> Tensor<T>::subtract(const Tensor<T>& other) const;
template <typename T>
Tensor<T> Tensor<T>::subtract(const T& scalar) const;

template <typename T>
Tensor<T> Tensor<T>::multiply(const Tensor<T>& other) const;
template <typename T>
Tensor<T> Tensor<T>::multiply(const T& scalar) const;

template <typename T>
Tensor<T> Tensor<T>::divide(const Tensor<T>& other) const;
template <typename T>
Tensor<T> Tensor<T>::divide(const T& scalar) const;



// Element Wise operations

template <typename T>
Tensor<T> Tensor<T>::negate() const;               // element-wise negation (-x)

template <typename T>
Tensor<T> Tensor<T>::abs() const;                  // element-wise absolute value

template <typename T>
Tensor<T> Tensor<T>::sqrt() const;                 // element-wise square root

template <typename T>
Tensor<T> Tensor<T>::exp() const;                  // element-wise exponentiation

template <typename T>
Tensor<T> Tensor<T>::log() const;                  // element-wise natural logarithm

template <typename T>
Tensor<T> Tensor<T>::sigmoid() const;              // element-wise sigmoid

template <typename T>
Tensor<T> Tensor<T>::tanh() const;                 // element-wise hyperbolic tangent






// Linear Algrbra Operations
template <typename T>
Tensor<T> Tensor<T>::matmul(const Tensor<T>& other) const;  // matrix multiplication (2D tensors)

template <typename T>
Tensor<T> Tensor<T>::transpose(int dim1, int dim2) const;   // transpose two dimensions

template <typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<int>& new_shape) const;  // reshape tensor

template <typename T>
Tensor<T> Tensor<T>::sum(int dim) const;                     // sum over a dimension

template <typename T>
Tensor<T> Tensor<T>::mean(int dim) const;                    // mean over a dimension

template <typename T>
Tensor<T> Tensor<T>::max(int dim) const;                     // max over a dimension

template <typename T>
Tensor<T> Tensor<T>::min(int dim) const;                     // min over a dimension

template <typename T>
Tensor<T> Tensor<T>::argmax(int dim) const;                  // index of max value along dim

template <typename T>
Tensor<T> Tensor<T>::argmin(int dim) const;                  // index of min value along dim




// Element-Wise comparison

template <typename T>
Tensor<bool> Tensor<T>::equal(const Tensor<T>& other) const;

template <typename T>
Tensor<bool> Tensor<T>::greater(const Tensor<T>& other) const;

template <typename T>
Tensor<bool> Tensor<T>::less(const Tensor<T>& other) const;

template <typename T>
Tensor<bool> Tensor<T>::greater_equal(const Tensor<T>& other) const;

template <typename T>
Tensor<bool> Tensor<T>::less_equal(const Tensor<T>& other) const;




// Util functions

template <typename T>
Tensor<T> Tensor<T>::broadcast_to(const std::vector<int>& new_shape) const;

template <typename T>
Tensor<T> Tensor<T>::slice(const std::vector<int>& start_indices, const std::vector<int>& sizes) const;

template <typename T>
Tensor<T> Tensor<T>::gather(int dim, const Tensor<int>& indices) const;

template <typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<int>& new_shape) const;
