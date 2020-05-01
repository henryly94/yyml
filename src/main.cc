#include <iostream>
#include "dense_layer.h"
#include "nn.h"
#include "op.h"
#include "tensor.h"
#include "variable.h"

class MyNN : public NN {
 public:
  MyNN() {
    TensorShape w_shape1{20, 256}, b_shape1{1, 256};
    TensorShape w_shape2{256, 16}, b_shape2{1, 16};
    set_layer("dense1", w_shape1, b_shape1);
    set_layer("dense2", w_shape2, b_shape2);
  }

  Variable<double>* Forward(Variable<double>* input) override {
    auto* o1 = (*get_layer(std::string("dense1")))(input);
    auto* o2 = (*get_layer(std::string("dense2")))(input);
    return o2;
  }
};

int main() {
  MyNN mynn;
  TensorShape input_shape{1, 20};
  Variable<double> nn_input(input_shape);
  auto* nn_output = mynn(&nn_input);
  mynn.print();
  std::cout << nn_input << '\n' << nn_output << '\n';
  Backward<double>(nn_output);
  std::cout << nn_input << '\n';
  mynn.print();
}
