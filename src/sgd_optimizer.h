#ifndef SGD_OPTIMIZER_H
#define SGD_OPTIMIZER_H

#include <vector>
#include "optimizer_interface.h"
#include "variable.h"

class SGDOptimizer : public OptimizerInterface {
 public:
  SGDOptimizer(std::vector<Variable<double>*> parameters, double learning_rate)
      : OptimizerInterface(parameters), learning_rate_(learning_rate) {}

  void step() override {
    for (auto* parameter : parameters_) {
      for (size_t i = 0; i < parameter->values_.total(); i++) {
        parameter->values_.data_[i] +=
            learning_rate_ * parameter->grads_.data_[i];
      }
    }
  }

 private:
  double learning_rate_;
};

#endif  // SGD_OPTIMIZER_H
