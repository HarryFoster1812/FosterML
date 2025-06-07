import torch

def test_ultimate_matrixmul_broadcast():
    print("=== Ultimate Test: Matrix Multiply with Broadcasting + Elementwise Ops ===")

    # Step 1: Define inputs
    A_data = [
        0.1, 0.2
    ]

    A = torch.tensor(A_data, dtype=torch.float32, requires_grad=True).reshape(2, 1)
    A.retain_grad()
    B = torch.full((3, 1, 2), 0.1, dtype=torch.float32, requires_grad=True)
    B.retain_grad()

    # Step 2: Computation graph
    D = torch.matmul(A, B)       # Shape: [2, 3, 3, 5]
    E = D + 0.1                  # scalar add
    F = E * D                    # elementwise multiply
    # G = torch.sigmoid(F)        # sigmoid
    # H = 1.0 - G                  # negate + 1
    # I = torch.tanh(H)           # tanh
    # J = (I - 0.0) / 1.0          # identity operation
    # K = torch.exp(J)            # exp

    print("RESULT OF CALCULATION")
    print(F)

    # Step 3: Backprop
    F.sum().backward()          # Equivalent to calling engine.backward(K)

    # Step 4: Print outputs

    print("\nGradient w.r.t. A:")
    print(A.grad)

    print("\nGradient w.r.t. B:")
    print(B.grad)

if __name__ == "__main__":
    test_ultimate_matrixmul_broadcast()
