#include <fstream>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>
#include "data_loader.h"
#include "dense_layer.h"
#include "function.h"
#include "nn.h"
#include "op.h"
#include "sgd_optimizer.h"
#include "tensor.h"
#include "variable.h"

class MyNN : public NN {
 public:
  MyNN() {
    TensorShape w_shape1{1, 16}, b_shape1{1, 16};
    TensorShape w_shape2{16, 32}, b_shape2{1, 32};
    TensorShape w_shape3{32, 16}, b_shape3{1, 16};
    TensorShape w_shape4{16, 1}, b_shape4{1, 1};
    SetLayer("dense1", w_shape1, b_shape1);
    SetLayer("dense2", w_shape2, b_shape2);
    SetLayer("dense3", w_shape3, b_shape3);
    SetLayer("dense4", w_shape4, b_shape4);
  }

  Variable<double>* Forward(Variable<double>* input) override {
    auto* o1 = GetLayer("dense1")(input);
    auto* a1 = ReLU(o1);
    auto* o2 = GetLayer("dense2")(a1);
    auto* a2 = ReLU(o2);
    auto* o3 = GetLayer("dense3")(a2);
    auto* a3 = ReLU(o3);
    auto* o4 = GetLayer("dense3")(a3);
    return o4;
  }
};

Variable<double>* LossFunc(Variable<double>* output, Variable<double>* label) {
  auto* diff = Substract<double>(label, output);
  return Multiply<double>(diff, diff);
}

void OutputResult(const std::vector<std::pair<double, double>>& x_and_ys) {
  std::ofstream out("result.txt");
  for (const auto& [x, y] : x_and_ys) {
    out << x << '\t' << y << '\n';
  }
  out.close();
}

int main() {
  MyNN mynn;
  SGDOptimizer optimizer(mynn.Parameters(), 0.001);
  TensorShape input_shape{1, 1}, output_shape{1, 1};

  DataLoader dataloader("data.txt", input_shape, output_shape);
  optimizer.Apply(RandomNumberGenerator::NormalDistribution);

  for (int i = 0; i < 200; i++) {
    double iter_loss = 0;
    for (auto& [nn_input, label] : dataloader) {
      optimizer.ZeroGrad();
      auto* nn_output = mynn(&nn_input);
      auto* loss = LossFunc(nn_output, &label);
      iter_loss += loss->values_.data_[0];
      loss->Backward();
      optimizer.Step();
    }
    std::cout << iter_loss << std::endl;
  }
  std::vector<std::pair<double, double>> x_and_ys;
  for (auto& [nn_input, _] : dataloader) {
    auto* nn_output = mynn(&nn_input);
    x_and_ys.emplace_back(nn_input.values_.data_[0],
                          nn_output->values_.data_[0]);
  }
  OutputResult(x_and_ys);
}
