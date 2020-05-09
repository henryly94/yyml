#ifndef OP_H
#define OP_H

#include <deque>
#include <functional>
#include <iostream>
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

//=====================================================

template <typename Type>
void AddBackward(Variable<Type>* va, Variable<Type>* vb,
                 Variable<Type>* result) {
  for (int i = 0; i < result->grads_.total(); i++) {
    /* dL      dL      d(upper)          dL
     * -- = -------- * ----------- =  -------- * 1
     * dx   d(upper)      dx          d(upper)
     *
     */
    va->grads_.data_[i] += result->grads_.data_[i];
    vb->grads_.data_[i] += result->grads_.data_[i];
  }
  Variable<Type>::factory::RemoveInstance(result);
}

template <typename Type>
void Add(Variable<Type>* va, Variable<Type>* vb, Variable<Type>* result) {
  result->autograd_.backward_fn = std::bind(AddBackward<Type>, va, vb, result);
  result->autograd_.next = {&va->autograd_, &vb->autograd_};
  for (size_t i = 0; i < va->values_.total(); i++) {
    result->values_.data_[i] = va->values_.data_[i] + vb->values_.data_[i];
  }
}

template <typename Type>
Variable<Type>* Add(Variable<Type>* va, Variable<Type>* vb) {
  auto* result = Variable<Type>::factory::GetNewInstance(va->values_.shape_);
  Add(va, vb, result);
  return result;
}

template <typename Type>
void SubstractBackward(Variable<Type>* va, Variable<Type>* vb,
                       Variable<Type>* result) {
  for (int i = 0; i < result->grads_.total(); i++) {
    /* dL      dL      d(upper)          dL
     * -- = -------- * ----------- =  -------- * 1
     * dx   d(upper)      dx          d(upper)
     *
     */
    va->grads_.data_[i] += result->grads_.data_[i];
    vb->grads_.data_[i] -= result->grads_.data_[i];
  }
  Variable<Type>::factory::RemoveInstance(result);
}

template <typename Type>
void Substract(Variable<Type>* va, Variable<Type>* vb, Variable<Type>* result) {
  result->autograd_.backward_fn =
      std::bind(SubstractBackward<Type>, va, vb, result);
  result->autograd_.next = {&va->autograd_, &vb->autograd_};
  for (size_t i = 0; i < va->values_.total(); i++) {
    result->values_.data_[i] = va->values_.data_[i] - vb->values_.data_[i];
  }
}

template <typename Type>
Variable<Type>* Substract(Variable<Type>* va, Variable<Type>* vb) {
  auto* result = Variable<Type>::factory::GetNewInstance(va->values_.shape_);
  Substract(va, vb, result);
  return result;
}

template <typename Type>
void MMBackward(Variable<Type>* va, Variable<Type>* vb,
                Variable<Type>* result) {
  size_t m = va->grads_.shape().dims[0], n = vb->grads_.shape().dims[1],
         p = va->grads_.shape().dims[1];
  for (size_t i = 0; i < m; i++) {
    for (size_t k = 0; k < p; k++) {
      for (size_t j = 0; j < n; j++) {
        va->grads_.data_[i * p + k] +=
            vb->values_.data_[k * n + j] * result->grads_.data_[i * n + j];
        vb->grads_.data_[k * n + j] +=
            va->values_.data_[i * p + k] * result->grads_.data_[i * n + j];
      }
    }
  }
  Variable<Type>::factory::RemoveInstance(result);
}

template <typename Type>
void MM(Variable<Type>* va, Variable<Type>* vb, Variable<Type>* result) {
  result->autograd_.backward_fn = std::bind(MMBackward<Type>, va, vb, result);
  result->autograd_.next = {&va->autograd_, &vb->autograd_};
  size_t m = va->values_.shape().dims[0], n = vb->values_.shape().dims[1],
         p = va->values_.shape().dims[1];
  for (size_t i = 0; i < m; i++) {
    for (size_t k = 0; k < p; k++) {
      for (size_t j = 0; j < n; j++) {
        result->values_.data_[i * n + j] +=
            va->values_.data_[i * p + k] * vb->values_.data_[k * n + j];
      }
    }
  }
}

template <typename Type>
Variable<Type>* MM(Variable<Type>* va, Variable<Type>* vb) {
  size_t m = va->values_.shape().dims[0], n = vb->values_.shape().dims[1],
         p = va->values_.shape().dims[1];
  TensorShape shape = {m, n};
  auto* result = Variable<Type>::factory::GetNewInstance(shape);
  MM(va, vb, result);
  return result;
}

template <typename Type>
void ReLUBackward(Variable<Type>* v, Variable<Type>* result) {
  for (size_t i = 0; i < v->values_.total(); i++) {
    v->grads_.data_[i] = v->values_.data_[i] >= 0 ? result->grads_.data_[i] : 0;
  }
}

template <typename Type>
void ReLU(Variable<Type>* v, Variable<Type>* result) {
  result->autograd_.backward_fn = std::bind(ReLUBackward<Type>, v, result);
  result->autograd_.next = {&v->autograd_};
  for (size_t i = 0; i < v->values_.total(); i++) {
    result->values_.data_[i] =
        v->values_.data_[i] >= 0 ? v->values_.data_[i] : 0;
  }
}

template <typename Type>
Variable<Type>* ReLU(Variable<Type>* v) {
  auto* result = Variable<Type>::factory::GetNewInstance(v->values_.shape_);
  ReLU(v, result);
  return result;
}

template <typename Type>
void MeanBackward(Variable<Type>* v, Variable<Type>* result) {
  Type total = v->grads_.total();
  for (size_t i = 0; i < v->grads_.total(); i++) {
    v->grads_.data_[i] += result->grads_.data_[0] / total;
  }
  Variable<Type>::factory::RemoveInstance(result);
}

template <typename Type>
void Mean(Variable<Type>* v, Variable<Type>* result) {
  result->autograd_.backward_fn = std::bind(MeanBackward<Type>, v, result);
  result->autograd_.next = {&v->autograd_};
  for (size_t i = 0; i < v->values_.total(); i++) {
    result->values_.data_[0] += v->values_.data_[i];
  }
  result->values_.data_[0] /= (Type)(v->values_.total());
}

template <typename Type>
Variable<Type>* Mean(Variable<Type>* v) {
  TensorShape scalar_shape({1});
  auto* result = Variable<Type>::factory::GetNewInstance(scalar_shape);
  Mean(v, result);
  return result;
}

template <typename Type>
void Backward(Variable<Type>* v) {
  for (int i = 0; i < v->grads_.total(); i++) {
    v->grads_.data_[i] = 1 / ((Type)v->grads_.total());
  }
  std::deque<Autograd<Type>*> bfs;
  bfs.push_back(&(v->autograd_));
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
