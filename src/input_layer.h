#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include "layer.h"
#include "vector.h"

struct InputLayer : public Layer {
  InputLayer(TensorShape<1> shape)
      : Layer(Layer::INPUT_LAYER), inputs_(shape) {}

  DoubleVector inputs_;
};

#endif  // INPUT_LAYER_H
