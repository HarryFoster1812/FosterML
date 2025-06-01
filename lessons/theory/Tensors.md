1. Computation Graph & Tensors

    A computation graph is a way to represent a function as a graph where each node is a tensor operation.

    In machine learning, neural networks and their computations are represented as computation graphs.

    Each tensor operation (like add, multiply, matmul) creates nodes that hold references to input tensors and the operation used.

    Forward pass computes the values; backward pass computes gradients by traversing the graph in reverse.

2. Tensors in Machine Learning vs. Math/Physics

    In math and physics, tensors are multi-dimensional arrays with specific transformation properties.

    In ML, tensors are primarily multi-dimensional arrays (n-dimensional arrays) for holding data and parameters.

    The core mathematical understanding required includes linear algebra (vectors, matrices), calculus (derivatives), and some tensor algebra.

3. Autodiff & Gradients

    Autodiff automatically computes gradients by chaining local derivatives through the computation graph.

    Each tensor stores:

        Its data.

        Whether it requires gradient (requires_grad).

        A pointer to its gradient tensor.

        References to its parent tensors and the backward function to compute gradients.

    During backpropagation:

        The loss tensor’s .backward() is called.

        Gradients propagate backward through the graph using stored backward functions.

    This setup allows efficient calculation of gradients without symbolic math or numerical approximation.

4. Tensor Class Design Considerations

    Use std::vector<T> internally for safe, dynamic storage of data.

    The tensor class should be a template to support float, double, or custom numeric types.

    Store shape, data, gradient, requires_grad flag, and backward function pointers.

    Implement operations like add, multiply, matmul, divide, subtract, and overload operators for syntactic convenience.

    Backward functions and parents are stored to support gradient computation.

    Returning new tensor objects after operations supports immutability and safer memory management.

5. Memory Management and References

    Use smart pointers (std::shared_ptr or std::unique_ptr) to manage tensor lifetimes.

    Returning references to internal smart pointers needs care to avoid dangling references.

    Immutable design pattern (returning new tensors) is safer and clearer.

    Optimizer holds references only to parameter tensors (weights/biases), not the entire layers.

    Optimizer steps update weights after gradients are computed, then zero out gradients for the next iteration.

6. Neural Network Flow (Visualized)

    Forward pass: Input → layers → output → loss

    Backward pass: Loss → output → layers → input, computing gradients along the way

    Each layer:

        Has parameters (tensors).

        Applies operations producing output tensors.

        Stores computation graph connections for backpropagation.

7. Activation Layers & Gradients

    Activation functions (like ReLU) are separate operations.

    Gradients for activations are computed by chaining backward functions.

    Gradient flow resembles a linked graph, not a simple list.

    Storing parents and backward functions is necessary to trace gradient flow properly.

8. Special Layers: Flatten and CNN

    Flatten: Reshapes tensors without changing data; backward reshapes gradients back.

    CNN layers: Perform convolutions in forward pass; backward pass computes gradients w.r.t inputs and filters using convolution transpose operations.

    These layers integrate into the computation graph like other tensor operations.
