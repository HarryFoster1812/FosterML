#pragma once
#include "../01 - Matrix/matrix.h"

template <typename T>
class ActivationFunction {
public:
   virtual Matrix<T> forward(const Matrix<T>& input) const;
   virtual Matrix<T> backward(const Matrix<T>& input,
                              const Matrix<T> gradient) const;

    static char[] name;
    
};

