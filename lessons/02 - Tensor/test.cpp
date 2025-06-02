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

template <typename T> void test_unary_operations() {
  std::cout << "\n=== Testing Unary Operations ===" << std::endl;

  Tensor<T> x({2, 2}, true);
  std::vector<T> data = {0.5, -0.5, 1.0, -1.0};
  x.setData(data);

  std::cout << "Original tensor:" << std::endl;
  x.print();

  // Test negate
  std::cout << "\nNegate:" << std::endl;
  Tensor<T> neg_x = x.negate();
  neg_x.print();

  // Test sqrt (only on positive values)
  Tensor<T> positive({2, 2}, true);
  std::vector<T> pos_data = {1.0, 4.0, 9.0, 16.0};
  positive.setData(pos_data);
  std::cout << "\nSquare root of [1, 4, 9, 16]:" << std::endl;
  Tensor<T> sqrt_x = positive.sqrt();
  sqrt_x.print();

  // Test tanh
  std::cout << "\nTanh:" << std::endl;
  Tensor<T> tanh_x = x.tanh();
  tanh_x.print();

  // Test sigmoid
  std::cout << "\nSigmoid:" << std::endl;
  Tensor<T> sig_x = x.sigmoid();
  sig_x.print();

  // Test exp
  std::cout << "\nExp:" << std::endl;
  Tensor<T> exp_x = x.exp();
  exp_x.print();
}

template <typename T> void test_reshape() {
  std::cout << "\n=== Testing Reshape ===" << std::endl;

  Tensor<T> original({2, 3}, false);
  std::vector<T> data = {1, 2, 3, 4, 5, 6};
  original.setData(data);

  std::cout << "Original shape [2, 3]:" << std::endl;
  original.print();

  std::cout << "\nReshaped to [3, 2]:" << std::endl;
  Tensor<T> reshaped = original.reshape({3, 2});
  reshaped.print();

  std::cout << "\nReshaped to [6]:" << std::endl;
  Tensor<T> flattened = original.reshape({6});
  flattened.print();

  std::cout << "\nReshaped to [1, 6]:" << std::endl;
  Tensor<T> row_vector = original.reshape({1, 6});
  row_vector.print();
}

template <typename T> void test_gradients() {
  std::cout << "\n=== Testing Gradients ===" << std::endl;

  // Simple gradient test: z = (x + y) * x
  Tensor<T> x({2, 2}, true);
  Tensor<T> y({2, 2}, true);

  std::vector<T> data_x = {2, 3, 4, 5};
  std::vector<T> data_y = {1, 1, 1, 1};
  x.setData(data_x);
  y.setData(data_y);

  std::cout << "X:" << std::endl;
  x.print();
  std::cout << "\nY:" << std::endl;
  y.print();

  // Forward pass: z = (x + y) * x
  Tensor<T> temp = x.add(y);      // temp = x + y
  Tensor<T> z = temp.multiply(x); // z = temp * x

  std::cout << "\nZ = (X + Y) * X:" << std::endl;
  z.print();

  // Set gradient of output to 1
  Tensor<T> grad_output({2, 2}, false);
  std::vector<T> ones = {1, 1, 1, 1};
  grad_output.setData(ones);
  // z.setGrad(grad_output);
  //
  // // Backward pass
  //
  // z.backwards();
  //
  // std::cout << "\nGradient of X (should be 2*x + y):" << std::endl;
  // x.getGrad().print();
  //
  // std::cout << "\nGradient of Y (should be x):" << std::endl;
  // y.getGrad().print();
}

template <typename T> void test_broadcasting() {
  std::cout << "\n=== Testing Broadcasting ===" << std::endl;

  // Test scalar + matrix
  Tensor<T> matrix({2, 3}, true);
  Tensor<T> scalar({1}, true);

  std::vector<T> mat_data = {1, 2, 3, 4, 5, 6};
  std::vector<T> scal_data = {10};
  matrix.setData(mat_data);
  scalar.setData(scal_data);

  std::cout << "Matrix:" << std::endl;
  matrix.print();
  std::cout << "\nScalar:" << std::endl;
  scalar.print();

  std::cout << "\nMatrix + Scalar (broadcasted):" << std::endl;
  Tensor<T> result = matrix.add(scalar);
  result.print();

  // Test row vector + matrix
  Tensor<T> row({1, 3}, true);
  std::vector<T> row_data = {100, 200, 300};
  row.setData(row_data);

  std::cout << "\nRow vector:" << std::endl;
  row.print();

  std::cout << "\nMatrix + Row (broadcasted):" << std::endl;
  Tensor<T> result2 = matrix.add(row);
  result2.print();
}

int main() {
  try {
    std::cout << "Starting Tensor Library Tests\n" << std::endl;

    test_basic_operations<double>();
    test_unary_operations<double>();
    test_reshape<double>();
    test_gradients<double>();
    test_broadcasting<double>();

    std::cout << "\n=== All tests completed successfully! ===" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
