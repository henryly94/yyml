#ifndef LAYER_H
#define LAYER_H

#include "vector.h"

struct Layer {
  enum LayerType { INPUT_LAYER, DENSE_LAYER };
  Layer(LayerType type) : type_(type) {}

  Layer* prev_layer_;
  LayerType type_;
};

#endif  // LAYER_H
