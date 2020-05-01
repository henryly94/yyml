#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer_interface.h"
#include "op.h"
#include "variable.h"

class DenseLayer : public LayerInterface {
 public:
  DenseLayer(TensorShape weight_shape, TensorShape bias_shape)
      : weight_(weight_shape), bias_(bias_shape) {}

  Variable<double>* operator()(Variable<double>* input) override {
    auto* product = MM<double>(input, &weight_);
    auto* out = Add<double>(product, &bias_);
    return out;
  }

  Variable<double> weight_;
  Variable<double> bias_;
};

#endif  // DENSE_LAYER_H
