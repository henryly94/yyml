#ifndef OPTIMIZER_INTERFACE_H
#define OPTIMIZER_INTERFACE_H

#include "variable.h"

class OptimizerInterface {
 public:
  OptimizerInterface(std::vector<Variable<double>*> parameters)
      : parameters_(parameters) {}

  virtual void step() = 0;

  void zero_grad() {
    for (auto* parameter : parameters_) {
      for (size_t i = 0; i < parameter->grads_.total(); i++) {
        parameter->grads_.data_[i] = 0;
      }
    }
  }

 protected:
  std::vector<Variable<double>*> parameters_;
};

#endif  // OPTIMIZER_INTERFACE_H
