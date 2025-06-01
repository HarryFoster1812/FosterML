#pragma once
#include <stdio.h>
#include <vector>

/**
 * @brief Template class representing a mathematical matrix.
 *
 * Provides basic matrix operations such as addition, subtraction,
 * multiplication (both scalar and matrix), transpose, determinant calculation,
 * and methods for generating special matrices (identity, random).
 *
 * @tparam T The type of the elements stored in the matrix.
 */
template <typename T> class Matrix {
  private:
    int rows; ///< Number of rows in the matrix
    int cols; ///< Number of columns in the matrix
    std::vector<T>
        data; ///< Flat vector storing matrix elements in row-major order

  public:
    /**
     * @brief Default constructor creates an empty matrix (0x0).
     */
    Matrix() : rows(0), cols(0) {}

    /**
     * @brief Constructs a matrix with given dimensions, initializing all
     * elements with initVal.
     *
     * @param rows Number of rows
     * @param cols Number of columns
     * @param initVal Initial value for all elements (default constructed if not
     * provided)
     */
    Matrix(int rows, int cols, const T& initVal = T());

    /// Copy constructor (defaulted)
    Matrix(const Matrix& other) = default;

    /// Move constructor (defaulted)
    Matrix(Matrix&& other) noexcept = default;

    /// Copy assignment operator (defaulted)
    Matrix& operator=(const Matrix& other) = default;

    /// Move assignment operator (defaulted)
    Matrix& operator=(Matrix&& other) noexcept = default;

    /// Destructor (defaulted)
    ~Matrix() = default;

    /**
     * @brief Access element at specified position (non-const).
     *
     * @param row Row index (0-based)
     * @param col Column index (0-based)
     * @return Reference to the element at (row, col)
     * @throws std::out_of_range if indices are out of bounds
     */
    T& operator()(int row, int col);

    /**
     * @brief Access element at specified position (const).
     *
     * @param row Row index (0-based)
     * @param col Column index (0-based)
     * @return Const reference to the element at (row, col)
     * @throws std::out_of_range if indices are out of bounds
     */
    const T& operator()(int row, int col) const;

    /**
     * @brief Matrix addition.
     *
     * @param other The matrix to add
     * @return A new matrix that is the sum of *this and other
     * @throws std::invalid_argument if dimensions do not match
     */
    Matrix<T> operator+(const Matrix<T>& other) const;

    /**
     * @brief Matrix subtraction.
     *
     * @param other The matrix to subtract
     * @return A new matrix that is the difference of *this and other
     * @throws std::invalid_argument if dimensions do not match
     */
    Matrix<T> operator-(const Matrix<T>& other) const;

    /**
     * @brief Matrix multiplication.
     *
     * @param other The matrix to multiply by
     * @return A new matrix representing the product of *this and other
     * @throws std::invalid_argument if inner dimensions do not match
     */
    Matrix<T> operator*(const Matrix<T>& other) const;

    /**
     * @brief Scalar multiplication.
     *
     * @param scalar The scalar value to multiply by
     * @return A new matrix where each element is multiplied by scalar
     */
    Matrix<T> operator*(const T& scalar) const;

    /**
     * @brief Transpose of the matrix.
     *
     * @return A new matrix that is the transpose of *this
     */
    Matrix<T> transpose() const;

    /**
     * @brief Calculate the determinant of the matrix.
     *
     * @return The determinant value
     * @throws std::logic_error if the matrix is not square
     */
    T& determinant() const;

    /**
     * @brief Generate an identity matrix of given size.
     *
     * @param size Size of the identity matrix (rows = cols = size)
     * @return Identity matrix of given size
     */
    static Matrix<T> identityMatrix(int size);

    /**
     * @brief Generate a random matrix with values between minValue and
     * maxValue.
     *
     * @param rows Number of rows
     * @param cols Number of columns
     * @param minValue Minimum random value
     * @param maxValue Maximum random value
     * @return A matrix filled with random values
     */
    static Matrix<T> randomMatrix(int rows, int cols, T minValue, T maxValue);

    /**
     * @brief Generate a random matrix using Gorat initialization.
     *
     * @param rows Number of rows
     * @param cols Number of columns
     * @param minValue Minimum random value
     * @param maxValue Maximum random value
     * @return A matrix initialized with the Gorat method
     */
    static Matrix<T> randomGorat(int rows, int cols, T minValue, T maxValue);

    /**
     * @brief Generate a random matrix using He initialization.
     *
     * @param rows Number of rows
     * @param cols Number of columns
     * @param minValue Minimum random value
     * @param maxValue Maximum random value
     * @return A matrix initialized with the He method
     */
    static Matrix<T> randomHe(int rows, int cols, T minValue, T maxValue);

    /**
     * @brief Print the matrix to the specified output stream.
     *
     * @param os Output stream (default: std::cout)
     */
    void print(std::ostream& os = std::cout) const;

    /**
     * @brief Get the number of rows.
     *
     * @return Number of rows in the matrix
     */
    int getRows() const { return rows; }

    /**
     * @brief Get the number of columns.
     *
     * @return Number of columns in the matrix
     */
    int getCols() const { return cols; }
};

#include "./Matrix.tpp"

// TODO:
// Add other random functions for NN initialisation (He and Xavier)
// Implement inverse helper
// Implement better constructor functions
// Implement / % ect
// Add resize
// Add insert col/row
// Implement assignment operators (inplace) += *= /=
//
// WARNING:
// NEED TO ADD DOCUMENTATION
//
//  NOTE:
//  static_assert(std::is_floating_point<T>::value, "randomXavier requires
//  floating point types"); constructor: std::initializer_list<T> constructor:
//  std::vector<T>
