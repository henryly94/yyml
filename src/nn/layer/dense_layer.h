#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <string>
#include <vector>
#include "base/factory.h"
#include "nn/layer/layer_interface.h"
#include "nn/op/op.h"
#include "nn/variable.h"

namespace yyml {
namespace nn {

class DenseLayer : public LayerInterface {
 public:
  using factory = Factory<DenseLayer>;
  DenseLayer(TensorShape weight_shape, TensorShape bias_shape, std::string name)
      : weight_(weight_shape, name + "/weight"),
        bias_(bias_shape, name + "/bias"),
        name_(name) {}

  Variable<double>* operator()(Variable<double>* input) override {
    auto* product = MM<double>(input, &weight_);
    auto* out = Add<double>(product, &bias_);
    return out;
  }

  std::vector<Variable<double>*> Parameters() override {
    return {&weight_, &bias_};
  }

  void print() override {
    std::cout << "Name:\n"
              << name_ << "\nWeight:\n"
              << weight_ << "\nBias: \n"
              << bias_ << std::endl;
  }

  Variable<double> weight_;
  Variable<double> bias_;
  std::string name_;
};

}  // namespace nn
}  // namespace yyml

#endif  // DENSE_LAYER_H
