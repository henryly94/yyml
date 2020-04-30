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
  for (size_t i = 0; i < va.total(); i++) {
    result.data_[i] = va.data_[i] + vb.data_[i];
  }
  return result;
}

template <typename Type>
void AddBackward(Tensor<Type>* a_grads, Tensor<Type>* b_grads,
                 const Tensor<Type>* result_grads) {
  for (int i = 0; i < result_grads->total(); i++) {
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
  for (size_t i = 0; i < va.values_.total(); i++) {
    result.values_.data_[i] = va.values_.data_[i] + vb.values_.data_[i];
  }
  return result;
}

template <typename Type>
void SubstractBackward(Tensor<Type>* a_grads, Tensor<Type>* b_grads,
                       const Tensor<Type>* result_grads) {
  for (size_t i = 0; i < result_grads->total(); i++) {
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
                        std::bind(SubstractBackward<Type>, &va.grads_,
                                  &vb.grads_, &result.grads_),
                        {&va.autograd_, &vb.autograd_});
  for (size_t i = 0; i < va.values_.total(); i++) {
    result.values_.data_[i] = va.values_.data_[i] - vb.values_.data_[i];
  }
  return result;
}

template <typename Type>
void MMBackward(Tensor<Type>* a_grads, Tensor<Type>* b_grads,
                Tensor<Type>* a_values, Tensor<Type>* b_values,
                const Tensor<Type>* result_grads) {
  size_t m = a_grads->shape().dims[0], n = b_grads->shape().dims[1],
         p = a_grads->shape().dims[1];
  for (size_t i = 0; i < m; i++) {
    for (size_t k = 0; k < p; k++) {
      for (size_t j = 0; j < n; j++) {
        a_grads->data_[i * p + k] +=
            b_values->data_[k * n + j] * result_grads->data_[i * n + j];
        b_grads->data_[k * n + j] +=
            a_values->data_[i * p + k] * result_grads->data_[i * n + j];
      }
    }
  }
}

template <typename Type>
Variable<Type> MM(Variable<Type>& va, Variable<Type>& vb) {
  size_t m = va.values_.shape().dims[0], n = vb.values_.shape().dims[1],
         p = va.values_.shape().dims[1];
  TensorShape shape = {m, n};
  Variable<Type> result(shape,
                        std::bind(MMBackward<Type>, &va.grads_, &vb.grads_,
                                  &va.values_, &vb.values_, &result.grads_),
                        {&va.autograd_, &vb.autograd_});
  for (size_t i = 0; i < m; i++) {
    for (size_t k = 0; k < p; k++) {
      for (size_t j = 0; j < n; j++) {
        result.values_.data_[i * n + j] +=
            va.values_.data_[i * p + k] * vb.values_.data_[k * n + j];
      }
    }
  }
  return result;
}

template <typename Type>
void MeanBackward(Tensor<Type>* grads, const Tensor<Type>* result_grads) {
  Type total = grads->total();
  for (size_t i = 0; i < grads->total(); i++) {
    grads->data_[i] += result_grads->data_[0] / total;
  }
}

template <typename Type>
Variable<Type> Mean(Variable<Type>& v) {
  TensorShape scalar_shape({1});
  Variable<Type> result(
      scalar_shape, std::bind(MeanBackward<Type>, &v.grads_, &result.grads_),
      {&v.autograd_});
  for (size_t i = 0; i < v.values_.total(); i++) {
    result.values_.data_[0] += v.values_.data_[i];
  }
  result.values_.data_[0] /= (Type)(v.values_.total());
  return result;
}

template <typename Type>
void Backward(Variable<Type>& v) {
  for (int i = 0; i < v.grads_.total(); i++) {
    v.grads_.data_[i] = 1 / ((Type)v.grads_.total());
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
