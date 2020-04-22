#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>

template <size_t N>
class TensorShape {
 public:
  size_t dims_[N];

  bool IsSameShape(const TensorShape<N> &other_shape) {
    for (size_t i = 0; i < N; i++) {
      if (dims_[i] != other_shape.dims_[i]) return false;
    }
    return true;
  }
};

template <size_t N, typename Type>
class Tensor {
 public:
  using shape_type = TensorShape<N>;

  Tensor(const shape_type &shape) : shape_(shape) {
    total_ = 1;
    for (size_t i = 0; i < N; i++) {
      total_ *= shape_.dims_[i];
    }
    buf_ = new Type[total_];
  }

  void print() {
    for (size_t i = 0; i < total_; i++) {
      std::cout << buf_[i] << ' ';
    }
    std::cout << '\n';
  }

  shape_type shape_;
  Type *buf_;
  size_t total_;
};

#endif  // TENSOR_H
