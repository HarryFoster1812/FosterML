#include "Tensor.hpp"
#include <cassert>
#include <iostream>
#include <vector>

template <typename T> void test_basic_operations() {
  std::cout << "=== Testing Basic Operations ===" << std::endl;

  // Create tensors
  Tensor<T> a({2, 3}, true); // 2x3 tensor with gradients
  Tensor<T> b({2, 3}, true);

  // Initialize with some values
  std::vector<T> data_a = {1, 2, 3, 4, 5, 6};
  std::vector<T> data_b = {7, 8, 9, 10, 11, 12};
  a.setData(data_a);
  b.setData(data_b);

  std::cout << "Tensor A:" << std::endl;
  a.print();

  std::cout << "\nTensor B:" << std::endl;
  b.print();

  // Test addition
  std::cout << "\nA + B:" << std::endl;
  Tensor<T> c = a.add(b);
  c.print();
  std::cout << "\n AUTODIFF TEST:" << std::endl;
  AutoDiffEngine<T> engine;
  engine.backward(c); // Compute gradients

  std::cout << "\n GRAD A:" << std::endl;
  a.getGrad().print(); // Expected: Tensor(2.0, shape=1) (1.0 + 1.0)
  std::cout << "\n GRAD B:" << std::endl;
  b.getGrad().print(); // Expected: Tensor(1.0, shape=1)

  std::cout << "\n=== Broadcast Backpropagation via AutoDiff ===" << std::endl;

  // Tensor A: shape (3)
  Tensor<T> d({3}, true);
  d.setData({10, 20, 30});

  // Tensor B: shape (2, 3)
  Tensor<T> e({2, 3}, true);
  e.setData({1, 2, 3, 4, 5, 6});

  // A is broadcasted to shape (2, 3), so A + B is shape (2, 3)
  Tensor<T> f = d.add(e); // Forward op with broadcasting

  std::cout << "Forward Result (D + E):\n";
  f.print();

  // Trigger backward pass
  engine.backward(f); // dL/dc = 1 by default

  std::cout << "\nGradient wrt D (should be shape {3} and values {2,2,2}):\n";
  d.getGrad().print();

  std::cout << "\nExpected Gradient for D:\n[2, 2, 2]\n";

  std::cout << "\nGradient wrt E (shouldebe shape {2,3} and values all 1):\n";
  e.getGrad().print();

  std::cout << "\nExpected Gradient for E:\n[1, 1, 1,\n 1, 1, 1]\n";

  // Test subtraction
  // std::cout << "\nA - B:" << std::endl;
  // Tensor<T> d = a.subtract(b);
  // d.print();
  //
  // // Test multiplication
  // std::cout << "\nA * B:" << std::endl;
  // Tensor<T> e = a.multiply(b);
  // e.print();
  //
  // // Test division
  // std::cout << "\nA / B:" << std::endl;
  // Tensor<T> f = a.divide(b);
  // f.print();
}

template <typename T> void test_broadcast1() {
  std::cout << "\n=== Test 2: Broadcast (1, 3) -> (2, 3) ===" << std::endl;

  Tensor<T> a({1, 3}, true);
  a.setData({10, 20, 30});

  Tensor<T> b({2, 3}, true);
  b.setData({1, 1, 1, 2, 2, 2});

  Tensor<T> c = a.add(b); // a is broadcast to (2, 3)

  AutoDiffEngine<T> engine;
  engine.backward(c);

  std::cout << "Gradient wrt A (expected: [[2, 2, 2]]):" << std::endl;
  a.getGrad().print(); // sum over axis 0: [1+1, 1+1, 1+1] = [2, 2, 2]

  std::cout << "Gradient wrt B (expected: all ones):" << std::endl;
  b.getGrad().print();
}

template <typename T> void test_scalar_broadcast() {

  std::cout << "\n=== Test 3: Broadcast scalar -> (2, 3) ===" << std::endl;

  Tensor<T> a({}, true); // scalar
  a.setData({5});

  Tensor<T> b({2, 3}, true);
  b.setData({1, 2, 3, 4, 5, 6});

  Tensor<T> c = a.add(b); // a is broadcast as scalar to (2, 3)

  AutoDiffEngine<T> engine;
  engine.backward(c);

  std::cout << "Gradient wrt A (expected: 6):" << std::endl;
  a.getGrad().print(); // 2×3 = 6 ones

  std::cout << "Gradient wrt B (expected: all ones):" << std::endl;
  b.getGrad().print();
}

template <typename T> void multiLayeredTest() {

  std::cout << "\n=== Multi-layered Broadcast + Backprop Test ===\n";

  Tensor<T> A({2, 3}, true);
  Tensor<T> B({1, 3}, true);
  Tensor<T> scalar({}, true); // scalar

  A.setData({1, 2, 3, 4, 5, 6});
  B.setData({10, 20, 30});
  scalar.setData({100});

  // C = A + B  (broadcast)
  Tensor<T> C = A.add(B);

  // D = C + scalar (scalar broadcast)
  Tensor<T> D = C.add(scalar);

  // E = D + A
  Tensor<T> E = D.add(A);

  // Backprop
  AutoDiffEngine<T> engine;
  engine.backward(E);

  // Print outputs
  std::cout << "\nA.grad (expected [[2,2,2],[2,2,2]]):" << std::endl;
  A.getGrad().print();

  std::cout << "\nB.grad (expected [[2,2,2]]):" << std::endl;
  B.getGrad().print();

  std::cout << "\nScalar.grad (expected 6):" << std::endl;
  scalar.getGrad().print();
}


template <typename T>
void multi_operation_test(){
    std::cout << "\n=== High-Level Multilayered Autodiff Test ===" << std::endl;

    Tensor<T> A({2, 3}, true);
    Tensor<T> B({1, 3}, true);
    Tensor<T> scalar({}, true);  // scalar

    A.setData({1, 2, 3, 4, 5, 6});
    B.setData({10, 20, 30});
    scalar.setData({2});

    // Forward pass
    Tensor<T> C = A.add(B);          // Broadcast
    Tensor<T> D = C.multiply(scalar); // Scalar mult
    Tensor<T> E = D.add(A);          // Add A again

    E.print();  // Should be [[23, 46, 69], [32, 55, 78]]

    AutoDiffEngine<T> engine;
    engine.backward(E);

    // Check gradients
    std::cout << "\nA.grad (expected [[3,3,3],[3,3,3]]):" << std::endl;
    A.getGrad().print();

    std::cout << "\nB.grad (expected [[4,4,4]]):" << std::endl;
    B.getGrad().print();

    std::cout << "\nScalar.grad (expected 141):" << std::endl;
    scalar.getGrad().print();
}

template <typename T>
void subtract_autoidff_test(){
    std::cout << "\n=== Subtraction / Negate Autodiff Test ===\n";

    Tensor<T> A({2, 2}, true);
    Tensor<T> B({1, 2}, true);
    Tensor<T> scalar({}, true);

    A.setData({10, 20, 30, 40});
    B.setData({1, 2});
    scalar.setData({3});

    Tensor<T> C = A.subtract(B);        // Broadcast
    Tensor<T> D = C.subtract(scalar);   // Scalar broadcast
    Tensor<T> E = D.subtract(A);        // Gradient accumulation on A

    E.print();  // Should be [[-4, -5], [-4, -5]]

    AutoDiffEngine<T> engine;
    engine.backward(E);

    // Expected gradients:
    std::cout << "\nA.grad (expected [[0,0],[0,0]]):" << std::endl;
    A.getGrad().print();

    std::cout << "\nB.grad (expected [[-2,-2]]):" << std::endl;
    B.getGrad().print();

    std::cout << "\nScalar.grad (expected -4):" << std::endl;
    scalar.getGrad().print();
}



template <typename T>
void multi_operation_chain_test() {
    std::cout << "\n=== Complex Multilayered Autodiff Chain Test ===" << std::endl;

    Tensor<T> A({2, 2}, true);
    Tensor<T> B({1, 2}, true);
    Tensor<T> C({2, 2}, true);
    Tensor<T> scalar1({}, true);
    Tensor<T> scalar2({}, true);

    A.setData({1, 2, 3, 4});
    B.setData({10, 20});
    C.setData({5, 6, 7, 8});
    scalar1.setData({2});
    scalar2.setData({4});

    // Forward pass:
    Tensor<T> D = A.add(B);              // D = A + B → broadcast B: [[11, 22], [13, 24]]
    Tensor<T> E = D.multiply(C);         // E = D * C: [[55, 132], [91, 192]]
    Tensor<T> F = E.subtract(scalar1);   // F = E - 2: [[53, 130], [89, 190]]
    Tensor<T> G = F.divide(scalar2);     // G = F / 4: [[13.25, 32.5], [22.25, 47.5]]
    Tensor<T> H = G.add(A);              // H = G + A: [[14.25, 34.5], [25.25, 51.5]]

    H.print();  // Final output tensor

    AutoDiffEngine<T> engine;
    engine.backward(H);

    // Expected Gradients:

    // ∂H/∂A = 1 + (∂G/∂F)*(∂F/∂E)*(∂E/∂D)*(∂D/∂A)
    // Since each operation is element-wise:
    // ∂H/∂A = 1 + ((1/4) * 1 * C * 1) = 1 + (C / 4)

    // A.grad = [[1 + 5/4, 1 + 6/4], [1 + 7/4, 1 + 8/4]] = [[2.25, 2.5], [2.75, 3.0]]

    // B.grad = same as A.grad, summed over rows (due to broadcast)
    // B.grad = [A.grad(0,0) + A.grad(1,0), A.grad(0,1) + A.grad(1,1)]
    //         = [2.25 + 2.75, 2.5 + 3.0] = [5.0, 5.5]

    // C.grad = ∂H/∂G * ∂G/∂F * ∂F/∂E * ∂E/∂C
    // ∂H/∂G = 1, ∂G/∂F = 1/4, ∂F/∂E = 1, ∂E/∂C = D
    // So, C.grad = D / 4 = [[11/4, 22/4], [13/4, 24/4]] = [[2.75, 5.5], [3.25, 6.0]]

    // scalar1.grad = -1/4 * ones = sum(-1/4) * 4 = -1.0
    // scalar2.grad = -F / (scalar2^2) = -F / 16, sum of all elements
    // scalar2.grad = -sum(F) / 16 = -(53 + 130 + 89 + 190)/16 = -462/16 = -28.875

    std::cout << "\nA.grad (expected [[2.25, 2.5], [2.75, 3.0]]):" << std::endl;
    A.getGrad().print();

    std::cout << "\nB.grad (expected [[5.0, 5.5]]):" << std::endl;
    B.getGrad().print();

    std::cout << "\nC.grad (expected [[2.75, 5.5], [3.25, 6.0]]):" << std::endl;
    C.getGrad().print();

    std::cout << "\nScalar1.grad (expected -1.0):" << std::endl;
    scalar1.getGrad().print();

    std::cout << "\nScalar2.grad (expected -28.875):" << std::endl;
    scalar2.getGrad().print();
}

int main() {
  try {
    std::cout << "Starting Tensor Library Tests\n" << std::endl;

    test_basic_operations<double>();
    test_basic_operations<int>();
    test_broadcast1<int>();
    test_scalar_broadcast<int>();
    multiLayeredTest<int>();
    multi_operation_test<int>();
    subtract_autoidff_test<int>();
    multi_operation_chain_test<float>();

    std::cout << "\n=== All tests completed successfully! ===" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
