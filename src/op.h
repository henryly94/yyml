#ifndef OP_H
#define OP_H

#include <deque>
#include <functional>
#include "autograd.h"
#include "tensor.h"
#include "variable.h"

template <typename Type>
Tensor<Type> Add(Tensor<Type>& va, Tensor<Type>& vb) {
  Tensor<Type> result(va.shape_);
  for (int i = 0; i < va.shape_.total; i++) {
    result.data_[i] = va.data_[i] + vb.data_[i];
  }
  return result;
}

template <typename Type>
void AddBackward(Tensor<Type>* a_grads, Tensor<Type>* b_grads,
                 const Tensor<Type>* result_grads) {
  for (int i = 0; i < result_grads->shape_.total; i++) {
    /* dL      dL      d(upper)          dL
     * -- = -------- * ----------- =  -------- * 1
     * dx   d(upper)      dx          d(upper)
     *
     */
    a_grads->data_[i] += result_grads->data_[i];
    b_grads->data_[i] += result_grads->data_[i];
  }
}

template <typename Type>
Variable<Type> Add(Variable<Type>& va, Variable<Type>& vb) {
  Variable<Type> result(
      va.values_.shape_,
      std::bind(AddBackward<Type>, &(va.grads_), &(vb.grads_), &result.grads_),
      {&va.autograd_, &vb.autograd_});
  for (int i = 0; i < va.values_.shape_.total; i++) {
    result.values_.data_[i] = va.values_.data_[i] + vb.values_.data_[i];
  }
  return result;
}

template <typename Type>
void SubstractBackward(Tensor<Type>* a_grads, Tensor<Type>* b_grads,
                       const Tensor<Type>* result_grads) {
  for (int i = 0; i < result_grads->shape_.total; i++) {
    /* dL      dL      d(upper)          dL
     * -- = -------- * ----------- =  -------- * (+/-)1
     * dx   d(upper)      dx          d(upper)
     *
     */
    a_grads->data_[i] += result_grads->data_[i];
    b_grads->data_[i] -= result_grads->data_[i];
  }
}

template <typename Type>
Variable<Type> Substract(Variable<Type>& va, Variable<Type>& vb) {
  Variable<Type> result(va.values_.shape_,
                        std::bind(SubstractBackward<Type>, &(va.grads_),
                                  &(vb.grads_), &result.grads_),
                        {&va.autograd_, &vb.autograd_});
  for (int i = 0; i < va.values_.shape_.total; i++) {
    result.values_.data_[i] = va.values_.data_[i] - vb.values_.data_[i];
  }
  return result;
}

template <typename Type>
void Backward(Variable<Type>& v) {
  for (int i = 0; i < v.grads_.shape_.total; i++) {
    v.grads_.data_[i] = 1 / ((Type)v.grads_.shape_.total);
  }
  std::deque<Autograd<Type>*> bfs;
  bfs.push_back(&(v.autograd_));
  while (!bfs.empty()) {
    auto* autograd = bfs.front();
    bfs.pop_front();
    for (auto* next : autograd->next) {
      bfs.push_back(next);
    }
    if (autograd->backward_fn != nullptr) {
      autograd->backward_fn();
    }
  }
}

#endif  // OP_H
