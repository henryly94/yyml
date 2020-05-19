#ifndef VARIABLE_H
#define VARIABLE_H

#include <string>
#include "base/factory.h"
#include "base/tensor.h"
#include "nn/autograd.h"

namespace yyml {
namespace nn {

template <typename Type>
void Backward(Variable<Type>* v);

template <typename Type>
class Variable {
 public:
  using factory = Factory<Variable>;
  Variable(TensorShape shape)
      : values_(shape),
        grads_(shape),
        autograd_({}, this, nullptr),
        name_("") {}

  Variable(TensorShape shape, std::string name)
      : values_(shape),
        grads_(shape),
        autograd_({}, this, nullptr),
        name_(name) {}

  Variable(TensorShape shape, BackwardFunction<Type> backward_fn,
           std::initializer_list<Autograd<Type>*> l)
      : values_(shape),
        grads_(shape),
        autograd_(l, this, backward_fn),
        name_("") {}

  void Backward() { ::yyml::nn::Backward<Type>(this); }

  friend std::ostream& operator<<(std::ostream& os, const Variable& v) {
    os << "Name: " << v.name_ << "\nValue: " << v.values_
       << "\nGrad: " << v.grads_;
    return os;
  }

  friend std::ostream& operator<<(std::ostream& os, const Variable* v) {
    return (os << *v);
  }

  Tensor<Type> values_;
  Tensor<Type> grads_;
  Autograd<Type> autograd_;
  std::string name_;
};

}  // namespace nn
}  // namespace yyml

#endif  // VARIABLE_H
