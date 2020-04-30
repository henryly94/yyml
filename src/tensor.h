#ifndef TENSOR_H
#define TENSOR_H

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <vector>

struct TensorShape {
  TensorShape(std::initializer_list<size_t> dims)
      : dim(dims.size()),
        dims(dims),
        total(std::accumulate(dims.begin(), dims.end(), 1,
                              std::multiplies<size_t>())) {}
  size_t dim;
  size_t total;
  std::vector<size_t> dims;
};

template <typename Type>
class Tensor {
 public:
  Tensor(TensorShape shape) : shape_(shape), data_(new Type[shape.total]) {}
  ~Tensor() { delete[] data_; }

  Tensor(const Tensor<Type>& other)
      : shape_(other.shape_), data_(new Type[shape_.total]) {
    std::cout << "copy ctor\n";
    std::copy(other.data_, other.data_ + shape_.total, data_);
  }

  Tensor(Tensor<Type>&& other) : shape_(other.shape_), data_(other.data_) {
    std::cout << "move ctor\n";
    other.data_ = nullptr;
  }

  Tensor<Type>& operator=(Tensor<Type>& other) {
    std::cout << "copy assign\n";
    shape_ = other.shape_;
    if (this != &other) {
      delete[] data_;
      data_ = new Type[shape_.total];
      std::copy(other.data_, other.data_ + shape_.total, data_);
    }
    return *this;
  }

  Tensor<Type>& operator=(Tensor<Type>&& other) {
    std::cout << "move assign\n";
    shape_ = other.shape_;
    if (this != &other) {
      delete[] data_;
      data_ = other.data_;
      other.data_ = nullptr;
    }
    return *this;
  }

  // Generate a non-template operator<< for the Tensor.
  friend std::ostream& operator<<(std::ostream& os, const Tensor& v) {
    size_t i;
    for (i = 0; i < v.shape_.total - 1; i++) {
      os << v.data_[i] << ' ';
    }
    os << v.data_[i];
    return os;
  }

  TensorShape shape_;
  Type* data_;
};

#endif  // TENSOR_H
