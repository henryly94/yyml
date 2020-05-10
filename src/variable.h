#ifndef VARIABLE_H
#define VARIABLE_H

#include "autograd.h"
#include "factory.h"
#include "tensor.h"

template <typename Type>
void Backward(Variable<Type>* v);

template <typename Type>
class Variable {
 public:
  using factory = Factory<Variable>;
  Variable(TensorShape shape)
      : values_(shape), grads_(shape), autograd_({}, this, nullptr) {}

  Variable(TensorShape shape, BackwardFunction<Type> backward_fn,
           std::initializer_list<Autograd<Type>*> l)
      : values_(shape), grads_(shape), autograd_(l, this, backward_fn) {}

  void Backward() { ::Backward<Type>(this); }

  friend std::ostream& operator<<(std::ostream& os, const Variable& v) {
    os << "Value: " << v.values_ << "\nGrad: " << v.grads_;
    return os;
  }

  friend std::ostream& operator<<(std::ostream& os, const Variable* v) {
    return (os << *v);
  }

  Tensor<Type> values_;
  Tensor<Type> grads_;
  Autograd<Type> autograd_;
};

#endif  // VARIABLE_H
