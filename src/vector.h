#ifndef VECTOR_H
#define VECTOR_H

#include "tensor.h"

namespace detail {

class IntVector : public Tensor<1, int> {
 public:
  IntVector(const shape_type &shape) : Tensor<1, int>(shape) {
    std::cout << "IntVector Ctor!\n";
  }
  void print() {
    std::cout << "Int Vector!\n";
    Tensor<1, int>::print();
  }
};

template <typename Type>
struct VectorTypedef {
  typedef Tensor<1, Type> type;
};

template <>
struct VectorTypedef<int> {
  typedef IntVector type;
};

}  // namespace detail

template <typename Type>
using Vector = typename detail::VectorTypedef<Type>::type;

using IntVector = Vector<int>;
using DoubleVector = Vector<double>;

#endif  // VECTOR_H
