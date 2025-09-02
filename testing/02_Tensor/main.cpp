#include <cassert>
#include <core/AutoDiffEngine.hpp>
#include <core/Tensor/TensorPtr.hpp>
#include <iostream>
#include <memory>
#include <vector>

using namespace FosterML;

template <typename T> void test_basic_operations() {
    std::cout << "=== Testing Basic Operations ===" << std::endl;

    // Create tensors
    TensorPtr<T> a =
        TensorPtr<T>::create({2, 3}, true); // 2x3 tensor with gradients
    TensorPtr<T> b = TensorPtr<T>::create({2, 3}, true);

    // Initialize with some values
    std::vector<T> data_a = {1, 2, 3, 4, 5, 6};
    std::vector<T> data_b = {7, 8, 9, 10, 11, 12};
    a->setData(data_a);
    b->setData(data_b);

    std::cout << "Tensor A:" << std::endl;
    a->print();

    std::cout << "\nTensor B:" << std::endl;
    b->print();

    // Test addition
    std::cout << "\nA + B:" << std::endl;
    TensorPtr<T> c = a->add(b);
    c->print();
    std::cout << "\n AUTODIFF TEST:" << std::endl;
    AutoDiffEngine<T> engine;
    engine.backward(c); // Compute gradients

    std::cout << "\n GRAD A:" << std::endl;
    a->getGrad()->print(); // Expected: Tensor(2.0, shape=1) (1.0 + 1.0)
    std::cout << "\n GRAD B:" << std::endl;
    b->getGrad()->print(); // Expected: Tensor(1.0, shape=1)

    std::cout << "\n=== Broadcast Backpropagation via AutoDiff ==="
              << std::endl;

    // Tensor A: shape (3)
    TensorPtr<T> d = TensorPtr<T>::create({3}, true);
    d->setData({10, 20, 30});

    // Tensor B: shape (2, 3)
    TensorPtr<T> e = TensorPtr<T>::create({2, 3}, true);
    e->setData({1, 2, 3, 4, 5, 6});

    // A is broadcasted to shape (2, 3), so A + B is shape (2, 3)
    TensorPtr<T> f = d->add(e); // Forward op with broadcasting

    std::cout << "Forward Result (D + E):\n";
    f->print();

    // Trigger backward pass
    engine.backward(f); // dL/dc = 1 by default

    std::cout << "\nGradient wrt D (should be shape {3} and values {2,2,2}):\n";
    d->getGrad()->print();

    std::cout << "\nExpected Gradient for D:\n[2, 2, 2]\n";

    std::cout << "\nGradient wrt E (shouldebe shape {2,3} and values all 1):\n";
    e->getGrad()->print();

    std::cout << "\nExpected Gradient for E:\n[1, 1, 1,\n 1, 1, 1]\n";

    // Test subtraction
    std::cout << "\nA - B:" << std::endl;
    TensorPtr<T> g = a->subtract(b);
    g->print();

    // Test multiplication
    std::cout << "\nA * B:" << std::endl;
    TensorPtr<T> h = a->multiply(b);
    h->print();

    // Test division
    std::cout << "\nA / B:" << std::endl;
    TensorPtr<T> i = a->divide(b);
    i->print();
}

template <typename T> void test_ultimate_matrixmul_broadcast() {
    std::cout << "\n=== Ultimate Test: Matrix Multiply with Broadcasting + "
                 "Elementwise Ops ==="
              << std::endl;

    // Create small-valued A to avoid saturation in sigmoid/tanh
    TensorPtr<T> A = TensorPtr<T>::create({2, 1}, true);
    A->setData({
        0.1, // batch 0
        0.2  // batch 1
    });
    A->setDebugName("A");

    TensorPtr<T> B = TensorPtr<T>::create({3, 1, 2}, true);
    std::vector<T> B_data(3 * 1 * 2, static_cast<T>(0.1)); // fill with 0.1
    B->setData(B_data);
    B->setDebugName("B");
    // Build computation graph
    TensorPtr<T> D = A->matrixmul(B); // shape:
    D->setDebugName("D");

    TensorPtr<T> E = D->add(0.1); // scalar add
    E->setDebugName("E");
    TensorPtr<T> F = E->multiply(D); // elementwise multiply
    F->setDebugName("F");
    TensorPtr<T> G = F->sigmoid();                  // sigmoid
    TensorPtr<T> H = G->negate()->add(1.0);         // -G + 1
    TensorPtr<T> I = H->tanh();                     // tanh
    TensorPtr<T> J = I->subtract(0.0)->divide(1.0); // identity
    TensorPtr<T> K = J->exp();                      // exp

    std::cout << "RESULT OF CALCULATION:" << std::endl;
    K->print();
    // Run backward pass
    AutoDiffEngine<T> engine;
    engine.backward(K); // gradient of K is implicitly ones

    // Print some outputs and gradients for verification

    std::cout << "Gradient wrt A" << std::endl;
    A->getGrad()->print();

    std::cout << "Gradient wrt B" << std::endl;
    B->getGrad()->print();
}

int main() {
    try {
        std::cout << "Starting Tensor Library Tests\n" << std::endl;
        // test_basic_operations<double>();
        // test_basic_operations<int>();
        // test_broadcast1<int>();
        // test_scalar_broadcast<int>();
        // multiLayeredTest<int>();
        // multi_operation_test<int>();
        // subtract_autoidff_test<int>();
        // multi_operation_chain_test<float>();
        // autodiff_all_ops_test<long double>();
        // test_matmul_broadcast<double>();
        // test_autodiff_matrixmul<float>();
        test_ultimate_matrixmul_broadcast<double>();

        std::cout << "\n=== All tests completed successfully! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
