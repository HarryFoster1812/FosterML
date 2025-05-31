#include <random>

/**
 * @brief Constructs a matrix with the given number of rows and columns,
 *        initializing all elements with the specified initial value.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param initVal Initial value for all elements (default constructed if not
 * specified)
 */
template <typename T>
Matrix<T>::Matrix(int rows, int cols, const T& initVal)
    : rows(rows), cols(cols), data(rows * cols, initVal) {}

/**
 * @brief Access element at specified row and column (non-const).
 *
 * Checks bounds and returns a reference to the element.
 *
 * @param row Row index (0-based)
 * @param col Column index (0-based)
 * @return Reference to element at (row, col)
 * @throws std::invalid_argument if indices are out of bounds
 */
template <typename T> T& Matrix<T>::operator()(int row, int col) {
    if (row >= rows || col >= cols || row < 0 || col < 0)
        throw std::invalid_argument("Index out of bounds");
    return data[row * cols + col];
}

/**
 * @brief Access element at specified row and column (const).
 *
 * Checks bounds and returns a const reference to the element.
 *
 * @param row Row index (0-based)
 * @param col Column index (0-based)
 * @return Const reference to element at (row, col)
 * @throws std::invalid_argument if indices are out of bounds
 */
template <typename T> const T& Matrix<T>::operator()(int row, int col) const {
    if (row >= rows || col >= cols || row < 0 || col < 0)
        throw std::invalid_argument("Index out of bounds");
    return data[row * cols + col];
}

/**
 * @brief Matrix addition.
 *
 * Adds corresponding elements of two matrices of the same dimensions.
 *
 * @param other The matrix to add
 * @return New matrix representing the element-wise sum
 * @throws std::invalid_argument if matrix dimensions do not match
 */
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

/**
 * @brief Matrix subtraction.
 *
 * Subtracts corresponding elements of two matrices of the same dimensions.
 *
 * @param other The matrix to subtract
 * @return New matrix representing the element-wise difference
 * @throws std::invalid_argument if matrix dimensions do not match
 */
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

/**
 * @brief Matrix multiplication.
 *
 * Performs matrix multiplication where the number of columns of
 * the left matrix matches the number of rows of the right matrix.
 *
 * @param other The matrix to multiply by
 * @return New matrix representing the product
 * @throws std::invalid_argument if inner dimensions do not match
 */
template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
    if (cols != other.rows)
        throw std::invalid_argument(
            "Matrix multiplication is not defined for the given matrices");

    Matrix<T> result(rows, other.cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            T total = T(); // Initialize to zero value of T
            for (int k = 0; k < cols; ++k) {
                total += (*this)(i, k) * other(k, j);
            }
            result(i, j) = total;
        }
    }
    return result;
}

/**
 * @brief Scalar multiplication.
 *
 * Multiplies every element of the matrix by the scalar value.
 *
 * @param scalar The scalar to multiply by
 * @return New matrix where each element is scaled by scalar
 */
template <typename T> Matrix<T> Matrix<T>::operator*(const T& scalar) const {
    Matrix<T> result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = scalar * (*this)(i, j);
        }
    }
    return result;
}

/**
 * @brief Transpose of the matrix.
 *
 * Flips rows and columns, so element (i, j) becomes (j, i).
 *
 * @return New matrix that is the transpose of this matrix
 */
template <typename T> Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> result(cols, rows);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result(j, i) = (*this)(i, j);
    return result;
}

/**
 * @brief Creates an identity matrix of specified size.
 *
 * An identity matrix has 1s on the diagonal and 0s elsewhere.
 *
 * @param size Number of rows and columns (square matrix)
 * @return Identity matrix of given size
 */
template <typename T> Matrix<T> Matrix<T>::identityMatrix(int size) {
    Matrix<T> result(size, size, T()); // Initialize with zeros
    for (int i = 0; i < size; ++i)
        result(i, i) = T(1); // Set diagonal elements to 1
    return result;
}

/**
 * @brief Generates a random matrix with elements between minVal and maxVal.
 *
 * Supports integral and floating point types. Throws if T is non-numeric.
 * Uses a static random engine seeded once.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param minVal Minimum random value (inclusive)
 * @param maxVal Maximum random value (inclusive for integers, exclusive for
 * floats)
 * @return Matrix filled with random values
 * @throws std::invalid_argument if T is not numeric
 */
template <typename T>
Matrix<T> Matrix<T>::randomMatrix(int rows, int cols, T minVal, T maxVal) {
    Matrix<T> result(rows, cols, T());

    static std::random_device rd;
    static std::mt19937 gen(rd());

    if constexpr (std::is_integral<T>::value) {
        // Use uniform int distribution for integral types
        std::uniform_int_distribution<T> dist(minVal, maxVal);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result(i, j) = dist(gen);
    } else if constexpr (std::is_floating_point<T>::value) {
        // Use uniform real distribution for floating point types
        std::uniform_real_distribution<T> dist(minVal, maxVal);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result(i, j) = dist(gen);
    } else {
        throw std::invalid_argument("Type provided is non-numeric");
    }

    return result;
}

/**
 * @brief Prints the matrix to the provided output stream.
 *
 * Elements in a row are separated by tabs, rows are separated by new lines.
 *
 * @param os Output stream to print to (default is std::cout)
 */
template <typename T> void Matrix<T>::print(std::ostream& os) const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            os << (*this)(i, j) << "\t";
        os << "\n";
    }
}
