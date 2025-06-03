#include "Tensor.hpp" // Adjust to your header file name
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

  // Test subtraction
  std::cout << "\nA - B:" << std::endl;
  Tensor<T> d = a.subtract(b);
  d.print();

  // Test multiplication
  std::cout << "\nA * B:" << std::endl;
  Tensor<T> e = a.multiply(b);
  e.print();

  // Test division
  std::cout << "\nA / B:" << std::endl;
  Tensor<T> f = a.divide(b);
  f.print();
}

int main() {
  try {
    std::cout << "Starting Tensor Library Tests\n" << std::endl;

    test_basic_operations<double>();
    test_basic_operations<int>();

    std::cout << "\n=== All tests completed successfully! ===" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
