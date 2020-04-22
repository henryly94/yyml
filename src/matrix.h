#ifndef MATRIX_H
#define MATRIX_H

#include <functional>
#include "tensor.h"

namespace detail {

template <typename Type>
using Matrix = Tensor<2, Type>;

class DoubleMatrix : public Tensor<2, double> {
 public:
  using value_initializer_type = std::function<double(size_t, size_t)>;
  DoubleMatrix(const shape_type &shape,
               value_initializer_type value_initializer)
      : Tensor<2, double>(shape) {
    for (size_t i = 0; i < total_; i++) {
      buf_[i] = value_initializer(i / shape.dim[1], i % shape.dim[1]);
    }
  }

  void print() {
    for (size_t i = 0; i < shape_.dim[0]; i++) {
      for (size_t j = 0; j < shape_.dim[1]; j++) {
        std::cout << buf_[i * shape_.dim[1] + j] << ' ';
      }
      std::cout << '\n';
    }
  }
};

template <typename Type>
struct MatrixTypedef {
  typedef Tensor<2, Type> type;
};

template <>
struct MatrixTypedef<double> {
  typedef DoubleMatrix type;
};

}  // namespace detail

template <typename Type>
using Matrix = typename detail::MatrixTypedef<Type>::type;

using DoubleMatrix = Matrix<double>;

#endif  // MATRIX_H
