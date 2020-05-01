#ifndef LAYER_INTERFACE_H
#define LAYER_INTERFACE_H

#include "variable.h"

class LayerInterface {
 public:
  virtual Variable<double>* operator()(Variable<double>* input) = 0;
};

#endif  // LAYER_INTERFACE_H
