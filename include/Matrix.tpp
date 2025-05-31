#include <random>

template <typename T>
Matrix<T>::Matrix(int rows, int cols, const T& initVal)
    : rows(rows), cols(cols), data(rows * cols, initVal) {}

template <typename T> T& Matrix<T>::operator()(int row, int col) {
    if (row >= rows || col >= cols || row < 0 || col < 0)
        throw std::invalid_argument("Index out of bounds");
    return data[row * cols + col];
}

template <typename T> const T& Matrix<T>::operator()(int row, int col) const {
    if (row >= rows || col >= cols || row < 0 || col < 0)
        throw std::invalid_argument("Index out of bounds");
    return data[row * cols + col];
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix sizes must match.");

    Matrix<T> result(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result(i, j) = (*this)(i, j) + other(i, j);
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix sizes must match.");

    Matrix<T> result(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result(i, j) = (*this)(i, j) - other(i, j);
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
    if (cols != other.rows)
        throw std::invalid_argument(
            "Matrix multiplication is not defined for the given matrices");

    Matrix<T> result(rows, other.cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            T total = T();
            for (int k = 0; k < cols; ++k) {
                total += (*this)(i, k) * other(k, j);
            }
            result(i, j) = total;
        }
    }
    return result;
}

template <typename T> Matrix<T> Matrix<T>::operator*(const T& scalar) const {
    Matrix<T> result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = scalar * (*this)(i, j);
        }
    }
    return result;
}

template <typename T> Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> result(cols, rows);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result(j, i) = (*this)(i, j);
    return result;
}

template <typename T> Matrix<T> Matrix<T>::identityMatrix(int size) {
    Matrix<T> result(size, size, T());
    for (int i = 0; i < size; ++i)
        result(i, i) = T(1);
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::randomMatrix(int rows, int cols, T minVal, T maxVal) {
    Matrix<T> result(rows, cols, T());

    static std::random_device rd;
    static std::mt19937 gen(rd());

    if constexpr (std::is_integral<T>::value) { // check for integer type on
                                                // compile time
        std::uniform_int_distribution<T> dist(
            minVal, maxVal); // use an integer distribution
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result(i, j) = dist(gen);
    }

    else if constexpr (std::is_floating_point<T>::
                           value) { // check for floating types on compile time
        std::uniform_real_distribution<T> dist(minVal, maxVal);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result(i, j) = dist(gen);
    } else {
        throw std::invalid_argument("Type provided is non-numeric");
    }

    return result;
}

template <typename T> void Matrix<T>::print(std::ostream& os) const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            os << (*this)(i, j) << "\t";
        os << "\n";
    }
}
