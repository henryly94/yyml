#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "dense_layer.h"
#include "input_layer.h"
#include "math.h"

void Forward(const DenseLayer& prev_layer, DenseLayer& next_layer) {
  MatrixMultiplyVector<double>(next_layer.weights_, prev_layer.outputs_,
                               &(next_layer.outputs_));
}

void Forward(const InputLayer& prev_layer, DenseLayer& next_layer) {
  MatrixMultiplyVector<double>(next_layer.weights_, prev_layer.inputs_,
                               &(next_layer.outputs_));
}

#endif  // NEURAL_NETWORK_H
