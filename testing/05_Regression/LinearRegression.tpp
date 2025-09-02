#include "LinearRegression.hpp"
#include <stdexcept>

namespace FosterML {
template <typename T> TensorPtr<T>& LinearRegression<T>::fit() {
    // bounds check input and output data to make sure they are the same length
} // closed-form

template <typename T>
TensorPtr<T> LinearRegression<T>::predict(const TensorPtr<T>& input) const {
    // make sure the number of samples are the same in each
    std::vector<int> inputShape = InputData.getShape();
    std::vector<int> outputShape = OutputData.getShape();
    if (inputShape.size() != 2 || outputShape.size() != 2 ||
        inputShape[0] != outputShape[0]) {
        throw std::invalid_argument(
            "Both input and oupt shape should be a matrix "
            "with the same number of rows eg nxm and nxp");
    }

    // construct x_aug
    TensorPtr<T> X_aug = InputData.cat(
        1, 1); // adds an another column to the data which is initialised to 1

    (X_aug->pinverse())->matrixmul(outputShape);
}

template <typename T>
void LinearRegression<T>::exportModel(std::string fileDir) {}

template <typename T>
void LinearRegression<T>::loadModel(std::string fileDir) {}

} // namespace FosterML
