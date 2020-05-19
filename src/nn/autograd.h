#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <functional>
#include <vector>

namespace yyml {
namespace nn {

template <typename Type>
using BackwardFunction = std::function<void()>;

template <typename Type>
class Variable;

template <typename Type>
struct Autograd {
  Autograd(std::vector<Autograd*> next, Variable<Type>* variable_ptr,
           BackwardFunction<Type> backward_fn)
      : next(next), variable_ptr(variable_ptr), backward_fn(backward_fn) {}
  std::vector<Autograd*> next;
  Variable<Type>* variable_ptr;
  BackwardFunction<Type> backward_fn;
};

}  // namespace nn
}  // namespace yyml

#endif  // AUTOGRAD_H
