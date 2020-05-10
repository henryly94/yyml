#include <iostream>
#include "dense_layer.h"
#include "nn.h"
#include "op.h"
#include "sgd_optimizer.h"
#include "tensor.h"
#include "variable.h"

class MyNN : public NN {
 public:
  MyNN() {
    TensorShape w_shape1{2, 4}, b_shape1{1, 4};
    TensorShape w_shape2{4, 3}, b_shape2{1, 3};
    SetLayer("dense1", w_shape1, b_shape1);
    SetLayer("dense2", w_shape2, b_shape2);
  }

  Variable<double>* Forward(Variable<double>* input) override {
    auto* o1 = GetLayer("dense1")(input);
    auto* o2 = GetLayer("dense2")(input);
    return o2;
  }
};

int main() {
  MyNN mynn;
  SGDOptimizer optimizer(mynn.Parameters(), 0.1);
  TensorShape input_shape{1, 2}, output_shape{1, 3};
  Variable<double> nn_input(input_shape), label(output_shape);
  label.values_.data_[0] = 1;
  label.values_.data_[1] = 1;
  label.values_.data_[2] = 1;

  optimizer.zero_grad();
  auto* nn_output = mynn(&nn_input);
  auto* loss = Substract<double>(&label, nn_output);
  mynn.print();
  std::cout << nn_input << '\n' << nn_output << '\n' << loss << '\n';
  std::cout << "================\n";
  Backward<double>(loss);
  std::cout << nn_input << '\n';
  mynn.print();
  optimizer.step();
  std::cout << "================\n";
  mynn.print();
}
