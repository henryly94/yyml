#ifndef CONV2D_LAYER_H
#define CONV2D_LAYER_H

#include <string>
#include <vector>
#include "nn/layer/layer_interface.h"
#include "nn/op/op.h"
#include "nn/variable.h"

namespace yyml {
namespace nn {

class Conv2DLayer : public LayerInterface {
 public:
  using factory = Factory<Conv2DLayer>;
  Conv2DLayer(TensorShape kernel_shape, std::string name)
      : kernel_(kernel_shape, name + "/kernel"), name_(name) {}

  Variable<double>* operator()(Variable<double>* input) override {
    return Conv2D<double>(input, &kernel_);
  }

  std::vector<Variable<double>*> Parameters() override { return {&kernel_}; }

  void print() override {
    std::cout << "Name:\n" << name_ << "\nKernel:\n" << kernel_ << std::endl;
  }

  Variable<double> kernel_;
  std::string name_;
};

}  // namespace nn
}  // namespace yyml

#endif  // CONV2D_LAYER_H
