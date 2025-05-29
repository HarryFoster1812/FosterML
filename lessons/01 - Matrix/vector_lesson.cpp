#include <iostream>
#include <vector>
#include <stdexcept>

template <typename T>
class Matrix{
public:
    Matrix() : rows(0), cols(0) {} // Default constructor
    Matrix(int rows, int cols, const T& initVal = T()); // Parameterized constructor
    ~Matrix() = default; // Destructor

    Matrix(const Matrix& other) = default;
    Matrix(Matrix&& other) noexcept = default;
    Matrix& operator=(const Matrix& other) = default;
    Matrix& operator=(Matrix&& other) noexcept = default;

    // Element access
    T& operator()(int row, int col);
    const T& operator()(int row, int col) const;

    // Matrix operations
    Matrix<T> operator+(const Matrix<T>& other) const;
    // Matrix<T> operator-(const Matrix<T>& other) const;
    // Matrix<T> operator*(const Matrix<T>& other) const;
    // Matrix<T> operator*(const T& scalar) const;
    // Matrix<T> transpose() const;

    // Utility functions
    // int getRows() const;
    // int getCols() const;
    void print(std::ostream& os = std::cout) const;

private:
    int rows, cols;
    std::vector<std::vector<T>> data;
};

// Constructor
template <typename T>
Matrix<T>::Matrix(int rows, int cols, const T& initVal)
    : rows(rows), cols(cols), data(rows, std::vector<T>(cols, initVal)) {}

// Operators
template <typename T>
T& Matrix<T>::operator()(int row, int col) {
    if(row >= rows || col >= cols || col < 0 || row < 0){
        throw std::invalid_argument("Index out of bounds of matrix");
    }
    return data[row][col];
}

template <typename T>
const T& Matrix<T>::operator()(int row, int col) const {
    if(row >= rows || col >= cols || col < 0 || row < 0){
        throw std::invalid_argument("Index out of bounds of matrix");
    }
    return data[row][col];
}

template <typename T>
Matrix<T> Matrix::operator+(const Matrix<T>& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix sizes must match.");
    }
    Matrix<T> result(rows, cols);
    
    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            result(i,j) = data[i][j] + other(i,j);
        }
    }

    return result;
}


template <typename T>
void Matrix<T>::print(std::ostream& os = std::cout) const{
    for(int i=0; i<rows; ++i){
        for(int j=0; j<cols; ++j)
            std::cout << data[i][j] << "\t";
        std::cout << "\n";
    }
}

int main(int argc, char *argv[]) {
    Matrix<int> A(2,2,3);
    Matrix<int> B(2,2,2);
    Matrix<int> C = A+B;
    C.print();
    return 0;
}
