#ifndef TENSOR_VIEW_H
#define TENSOR_VIEW_H

#include "tensor.h"

template <size_t N, typename Type>
struct TensorView {
  using sub_tensor_type = TensorView<N - 1, Type>;

  TensorView(size_t dim[N], Type* ptr)
      : ptr(ptr), range(dim[0]), sub_tensor(dim + 1, ptr) {
    step = 1;
    for (size_t i = 1; i < N; i++) {
      step *= dim[i];
    }
  }

  sub_tensor_type operator[](size_t idx) {
    sub_tensor.ptr = ptr + idx * step;
    return sub_tensor;
  }
  Type* ptr;
  size_t step;
  size_t range;
  sub_tensor_type sub_tensor;
};

template <typename Type>
struct TensorView<1, Type> {
  TensorView(size_t dim[1], Type* ptr) : ptr(ptr), range(dim[0]) {}

  Type& operator[](size_t idx) { return *(ptr + idx); }

  Type* ptr;
  size_t range;
};

#endif  // TENSOR_VIEW_H
