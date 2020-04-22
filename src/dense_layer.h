#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer.h"
#include "matrix.h"
#include "vector.h"

struct DenseLayer : public Layer {
  DenseLayer(TensorShape<2> shape,
             DoubleMatrix::value_initializer_type initializer)
      : Layer(Layer::DENSE_LAYER),
        weights_(shape, initializer),
        outputs_(DoubleVector::shape_type{shape.dim[0]}) {}

  DoubleMatrix weights_;
  DoubleVector outputs_;
};

#endif  // DENSE_LAYER_H
