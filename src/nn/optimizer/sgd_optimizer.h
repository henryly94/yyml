#ifndef SGD_OPTIMIZER_H
#define SGD_OPTIMIZER_H

#include <vector>
#include "nn/optimizer/optimizer_interface.h"
#include "nn/variable.h"

namespace yyml {
namespace nn {

class SGDOptimizer : public OptimizerInterface {
 public:
  SGDOptimizer(std::vector<Variable<double>*> parameters, double learning_rate)
      : OptimizerInterface(parameters), learning_rate_(learning_rate) {}

  void Step() override {
    for (auto* parameter : parameters_) {
      for (size_t i = 0; i < parameter->values_.total(); i++) {
        parameter->values_.data_[i] -=
            learning_rate_ * parameter->grads_.data_[i];
      }
    }
  }

 private:
  double learning_rate_;
};

}  // namespace nn
}  // namespace yyml

#endif  // SGD_OPTIMIZER_H
