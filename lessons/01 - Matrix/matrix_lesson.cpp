#include "./matrix.cpp"

int main() {
    Matrix<int> A(2, 2, 0);
    A(0,0)=11;
    A(0,1)=3;
    A(1,0)=7;
    A(1,1)=11;

    std::cout << "Matrix A:\n";
    A.print();

    Matrix<int> B(2, 3, 0);
    B(0,0)=8;
    B(0,1)=0;
    B(0,2)=1;
    B(1,0)=0;
    B(1,1)=3;
    B(1,2)=5;

    std::cout << "Matrix B:\n";
    B.print();

    // Matrix<int> C = A + B;
    // std::cout << "A+B:\n";
    // C.print();

    Matrix<int> D = A * B;
    std::cout << "A*B:\n";
    D.print();


    Matrix<int> E = D.transpose();
    std::cout << "D^T:\n";
    E.print();


    Matrix<int> I = Matrix<int>::identityMatrix(7);
    std::cout << "identity Matrix Test:\n";
    I.print();

    Matrix<int> F = Matrix<int>::randomMatrix(2, 2, 0, 100);
    std::cout << "Random int Matrix Test:\n";
    F.print();

    auto G = Matrix<float>::randomMatrix(5, 5, 0, 100);
    std::cout << "Random double Matrix Test:\n";
    G.print();

    return 0;
}
