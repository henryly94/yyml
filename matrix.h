#ifndef MATRIX_H
#define MATRIX_H

#include "tensor.h"

namespace detail {

template <typename Type>
using Matrix = Tensor<2, Type>;

class DoubleMatrix : public Tensor<2, double> {
 public:
  DoubleMatrix(const shape_type &shape, double value)
      : Tensor<2, double>(shape) {
    for (size_t i = 0; i < total_; i++) {
      buf_[i] = value;
    }
  }

  void print() {
    std::cout << "Double Matrix!\n";
    Tensor<2, double>::print();
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
