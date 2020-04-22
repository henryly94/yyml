#ifndef MATH_H
#define MATH_H

#include "vector.h"

IntVector AddIntVector(IntVector &a, IntVector &b) {
  IntVector result(a.shape_);
  for (int i = 0; i < result.shape_.dim[0]; i++) {
    result.buf_[i] = a.buf_[i] + b.buf_[i];
  }
  return result;
}

#endif  // MATH_H
