#ifndef OPTIMIZER_INTERFACE_H
#define OPTIMIZER_INTERFACE_H

#include <functional>
#include "nn/variable.h"

namespace yyml {
namespace nn {

class OptimizerInterface {
 public:
  OptimizerInterface(std::vector<Variable<double>*> parameters)
      : parameters_(parameters) {}

  virtual void Step() = 0;

  void ZeroGrad() {
    for (auto* parameter : parameters_) {
      for (size_t i = 0; i < parameter->grads_.total(); i++) {
        parameter->grads_.data_[i] = 0;
      }
    }
  }

  void Apply(std::function<double()> init_func) {
    for (auto* parameter : parameters_) {
      for (size_t i = 0; i < parameter->grads_.total(); i++) {
        parameter->values_.data_[i] = init_func();
      }
    }
  }

 protected:
  std::vector<Variable<double>*> parameters_;
};

}  // namespace nn
}  // namespace yyml

#endif  // OPTIMIZER_INTERFACE_H
