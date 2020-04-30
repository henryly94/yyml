#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <functional>
#include <vector>
#include "tensor.h"

template <typename Type>
using BackwardFunction = std::function<void()>;

template <typename Type>
struct Autograd {
  Autograd(std::vector<Autograd*> next, Tensor<Type>* grad_ptr,
           BackwardFunction<Type> backward_fn)
      : next(next), grad_ptr(grad_ptr), backward_fn(backward_fn) {}
  std::vector<Autograd*> next;
  Tensor<Type>* grad_ptr;
  BackwardFunction<Type> backward_fn;
};

#endif  // AUTOGRAD_H
