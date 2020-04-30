#ifndef VARIABLE_H
#define VARIABLE_H

#include "autograd.h"
#include "tensor.h"

template <typename Type>
class Variable {
 public:
  Variable(TensorShape shape)
      : values_(shape), grads_(shape), autograd_({}, &grads_, nullptr) {}

  Variable(TensorShape shape, BackwardFunction<Type> backward_fn,
           std::initializer_list<Autograd<Type>*> l)
      : values_(shape), grads_(shape), autograd_(l, &grads_, backward_fn) {}

  friend std::ostream& operator<<(std::ostream& os, const Variable& v) {
    os << "Value: " << v.values_ << "\nGrad: " << v.grads_;
    return os;
  }

  Tensor<Type> values_;
  Tensor<Type> grads_;
  Autograd<Type> autograd_;
};

#endif  // VARIABLE_H
