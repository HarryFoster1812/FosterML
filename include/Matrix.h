#pragma once
#include <iostream>
#include <stdexcept>
#include <vector>

template <typename T> class Matrix {
  private:
    int rows, cols;
    std::vector<T> data;

  public:
    Matrix() : rows(0), cols(0) {}
    Matrix(int rows, int cols, const T& initVal = T());
    Matrix(const Matrix& other) = default;
    Matrix(Matrix&& other) noexcept = default;
    Matrix& operator=(const Matrix& other) = default;
    Matrix& operator=(Matrix&& other) noexcept = default;
    ~Matrix() = default;

    T& operator()(int row, int col);
    const T& operator()(int row, int col) const;

    Matrix<T> operator+(const Matrix<T>& other) const;
    Matrix<T> operator-(const Matrix<T>& other) const;
    Matrix<T> operator*(const Matrix<T>& other) const;
    Matrix<T> operator*(const T& scalar) const;

    Matrix<T> transpose() const;
    T& determinant() const;
    static Matrix<T> identityMatrix(int size);

    static Matrix<T> randomMatrix(int rows, int cols, T minValue, T maxValue);
    static Matrix<T> randomGorat(int rows, int cols, T minValue, T maxValue);
    static Matrix<T> randomHe(int rows, int cols, T minValue, T maxValue);

    void print(std::ostream& os = std::cout) const;

    int getRows() const { return rows; }
    int getCols() const { return cols; }
};

#include "Matrix.tpp"

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
