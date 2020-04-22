#ifndef MATH_H
#define MATH_H

#include "matrix.h"
#include "vector.h"

IntVector AddIntVector(IntVector& a, IntVector& b) {
  IntVector result(a.shape_);
  for (int i = 0; i < result.shape_.dim[0]; i++) {
    result.buf_[i] = a.buf_[i] + b.buf_[i];
  }
  return result;
}

template <typename Type>
void MatrixMultiplyVector(const Matrix<Type>& matrix,
                          const Vector<Type>& src_vector,
                          Vector<Type>* dst_vector) {
  for (int i = 0; i < matrix.shape_.dim[0]; i++) {
    dst_vector->buf_[i] = 0;
    for (int j = 0; j < matrix.shape_.dim[1]; j++) {
      dst_vector->buf_[i] +=
          matrix.buf_[i * matrix.shape_.dim[1] + j] * src_vector.buf_[j];
    }
  }
}

#endif  // MATH_H
