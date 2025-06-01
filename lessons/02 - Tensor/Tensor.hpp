#include <vector>

template <typename T>
class Tensor {
    friend class AutoDiffEngine;
private:
    std::vector<T> data;
    std::vector<int> shape;
    std::unique_ptr<Tensor<T>> gradient;
    bool requires_gradient;

    std::vector<shared_ptr<Tensor<T>>> parents; // the parents that were used to compute the tensor
    std::function<void()> backwardsFunction; // the function called to alter the parents gradients
public:
    // Constructors
    Tensor();

    const std::vector<int>& getShape() const { return shape; }
    const std::vector<T>& getData() const { return data; }
    const Tensor<T>& getGrad() const { return gradient; }
    const std::vector<shared_ptr<Tensor<T>>> getParents(){return parents;} 
    bool requiresGrad() const { return requires_gradient; }

    void zeroGrad() {
        if (requires_grad) {
            std::vector<T> gradientVector = gradient.getData();
            std::fill(gradientVector.begin(), gradientVector.end(), static_cast<T>(0));
        }
    }

    void setData(const std::vector<T>& new_data) {
        if (new_data.size() != data.size())
            throw std::runtime_error("Size mismatch in setData");
        data = new_data;
   }


    void setBackwardsFunction(std::function<void()> func){
        backwardsFunction = func;
    }

    Tensor<T>(Tensor<T> &&) = default;
    Tensor<T>(const Tensor<T> &) = default;
    Tensor<T>& operator=(Tensor<T> &&) = default;
    Tensor<T>& operator=(const Tensor<T> &) = default;
    ~Tensor() = default;

    // Core Arithmetic

    Tensor<T> add(const Tensor<T>& other) const;
    Tensor<T> add(const T& scalar) const;

    Tensor<T> subtract(const Tensor<T>& other) const;
    Tensor<T> subtract(const T& scalar) const;

    Tensor<T> multiply(const Tensor<T>& other) const;
    Tensor<T> multiply(const T& scalar) const;

    Tensor<T> divide(const Tensor<T>& other) const;
    Tensor<T> divide(const T& scalar) const;


    // Element Wise operations

    Tensor<T> negate() const;               // element-wise negation (-x)
    Tensor<T> abs() const;                  // element-wise absolute value
    Tensor<T> sqrt() const;                 // element-wise square root
    Tensor<T> exp() const;                  // element-wise exponentiation
    Tensor<T> log() const;                  // element-wise natural logarithm
    Tensor<T> sigmoid() const;              // element-wise sigmoid
    Tensor<T> tanh() const;                 // element-wise hyperbolic tangent
    
    // Linear Algrbra Operations
    Tensor<T> matmul(const Tensor<T>& other) const;  // matrix multiplication (2D tensors)
    Tensor<T> transpose(int dim1, int dim2) const;   // transpose two dimensions
    Tensor<T> reshape(const std::vector<int>& new_shape) const;  // reshape tensor
    Tensor<T> sum(int dim) const;                     // sum over a dimension
    Tensor<T> mean(int dim) const;                    // mean over a dimension
    Tensor<T> max(int dim) const;                     // max over a dimension
    Tensor<T> min(int dim) const;                     // min over a dimension
    Tensor<T> argmax(int dim) const;                  // index of max value along dim
    Tensor<T> argmin(int dim) const;                  // index of min value along dim

    // Element-Wise comparison
    Tensor<bool> equal(const Tensor<T>& other) const;
    Tensor<bool> greater(const Tensor<T>& other) const;
    Tensor<bool> less(const Tensor<T>& other) const;
    Tensor<bool> greater_equal(const Tensor<T>& other) const;
    Tensor<bool> less_equal(const Tensor<T>& other) const;

    Tensor<T> broadcast_to(const std::vector<int>& new_shape) const;
    Tensor<T> slice(const std::vector<int>& start_indices, const std::vector<int>& sizes) const;
    Tensor<T> gather(int dim, const Tensor<int>& indices) const;
    Tensor<T> reshape(const std::vector<int>& new_shape) const;
    
    /**
    * This is the top level function which should only be called on loss/output tensor
    */

    // Operator definitons for syntactic sugar
    Tensor<T> operator*(const Tensor<T>& other) const {return multiply(other);} // ADD SMART MULTIPLICATION (AUTO MATRIX MUL FOR 2D)
    Tensor<T> operator*(const T& scalar) const {return multiply(scalar);}

    Tensor<T> operator+(const Tensor<T>& other) const {return add(other);}
    Tensor<T> operator+(const T& scalar) const {return add(scalar);}

    Tensor<T> operator-(const Tensor<T>& other) const {return subtract(other);}
    Tensor<T> operator-(const T& scalar) const {return subtract(scalar);}

    Tensor<T> operator/(const Tensor<T>& other) const {return divide(other);}
    Tensor<T> operator/(const T& scalar) const {return divide(scalar);}

    // Batch Multiplication
    
    static bool canBatchMultiply(const Tensor<T>& tensorA, const Tensor<T>& tensorB);
    Tensor<T> batchMultiply(const Tensor<T>& other) const;

protected:
    void _backwards(){
        if(backwardsFunction)
            backwardsFunction();
    }
};


#include "Tensor.tpp"
