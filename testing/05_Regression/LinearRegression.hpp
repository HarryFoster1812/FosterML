#include <Tensor.hpp>

namespace FosterML {
template <typename T> class LinearRegression {
  private:
    // model will return
    TensorPtr<T> gradient;
    TensorPtr<T> InputData;
    TensorPtr<T> OutputData;

    // loss function

  public:
    LinearRegression<T>(const TensorPtr<T> input, const TensorPtr<T> output)
        : InputData(input), OutputData(output) {}

    TensorPtr<T>& fit(); // closed-form
    TensorPtr<T> predict(const TensorPtr<T>& input) const;

    T meanSquaredError(const TensorPtr<T>& X_test,
                       const TensorPtr<T>& Y_test) const;

    void exportModel(std::string fileDir);
    void loadModel(std::string fileDir);

    void setInput(const TensorPtr<T>& input) { InputData = input; }
    void setOutput(const TensorPtr<T>& output) { OutputData = output; }
};
} // namespace FosterML

#include <LinearRegression.tpp>
