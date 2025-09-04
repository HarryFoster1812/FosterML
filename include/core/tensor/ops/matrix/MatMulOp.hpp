#pragma once
#include <core/tensor/ops/base/OpNode.hpp>
#include <core/tensor/util.hpp>
#include <memory>
#include <vector>

namespace FosterML {

template <typename T> class MatMulOp : public OpNode<T> {
  public:
    MatMulOp(const TensorPtr<T>& A, const TensorPtr<T>& B)
        : OpNode<T>({A, B}, inferResultShape(A, B),
                    A->requiresGrad() || B->requiresGrad()) {}

    // Forward
    void forward() override {
        const auto& A = this->inputs[0];
        const auto& B = this->inputs[1];

        const auto& shapeA = A->getShape();
        const auto& shapeB = B->getShape();
        const auto& outShape = this->output->getShape();

        int M = shapeA[shapeA.size() - 2];
        int N = shapeA[shapeA.size() - 1];
        int P = shapeB[shapeB.size() - 1];

        std::vector<int> batchdimA(shapeA.begin(), shapeA.end() - 2);
        std::vector<int> batchdimB(shapeB.begin(), shapeB.end() - 2);

        // Batch dimensions
        std::vector<int> batchShape(outShape.begin(), outShape.end() - 2);

        // Broadcast A to [batch..., M, N]
        TensorPtr<T> A_broadcasted =
            (batchdimA == batchShape)
                ? A
                : A->broadcast_to(
                      concat_vec(batchShape, std::vector<int>{M, N}));

        TensorPtr<T> B_broadcasted =
            (batchdimB == batchShape)
                ? B
                : B->broadcast_to(
                      concat_vec(batchShape, std::vector<int>{N, P}));

        // Perform matmul
        std::vector<int> index(batchShape.size(), 0);
        do {
            for (int m = 0; m < M; ++m) {
                for (int p = 0; p < P; ++p) {
                    T sum = 0;
                    for (int n = 0; n < N; ++n) {
                        sum += (*A_broadcasted)(concat_vec(index, {m, n})) *
                               (*B_broadcasted)(concat_vec(index, {n, p}));
                    }
                    (*this->output)(concat_vec(index, {m, p})) = sum;
                }
            }
        } while (Tensor<T>::incrementIndex(index, batchShape));
    }

    // Backward
    void backward() override {
        auto grad_out = this->output->getGrad();
        const auto& A = this->inputs[0];
        const auto& B = this->inputs[1];

        if (A->requiresGrad()) {
            TensorPtr<T> B_T = B->transpose(-2, -1);
            TensorPtr<T> dA_raw = grad_out->matrixmul(B_T);
            TensorPtr<T> dA =
                Tensor<T>::sum_over_broadcasted_axes(dA_raw, A->getShape());
            A->addGrad(dA);
        }

        if (B->requiresGrad()) {
            TensorPtr<T> A_T = A->transpose(-2, -1);
            TensorPtr<T> dB_raw = A_T->matrixmul(grad_out);
            TensorPtr<T> dB =
                Tensor<T>::sum_over_broadcasted_axes(dB_raw, B->getShape());
            B->addGrad(dB);
        }
    }

  private:
    static std::vector<int> inferResultShape(const TensorPtr<T>& A,
                                             const TensorPtr<T>& B) {
        const auto& shapeA = A->getShape();
        const auto& shapeB = B->getShape();

        if (shapeA.size() < 2 || shapeB.size() < 2)
            throw std::invalid_argument("Both tensors must be at least 2D");

        int M = shapeA[shapeA.size() - 2];
        int N = shapeA[shapeA.size() - 1];
        int N2 = shapeB[shapeB.size() - 2];
        int P = shapeB[shapeB.size() - 1];

        if (N != N2)
            throw std::invalid_argument(
                "Inner dimensions do not match for matmul");

        // Broadcast batch dimensions
        std::vector<int> batchA(shapeA.begin(), shapeA.end() - 2);
        std::vector<int> batchB(shapeB.begin(), shapeB.end() - 2);
        std::vector<int> batch_shape =
            Tensor<T>::infer_broadcast_shape(batchA, batchB);

        std::vector<int> result_shape = batch_shape;
        result_shape.push_back(M);
        result_shape.push_back(P);
        return result_shape;
    }
};
} // namespace FosterML
