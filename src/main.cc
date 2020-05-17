#include <fstream>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>
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

std::vector<std::pair<double, double>> GetInput(std::string_view data_path) {
  std::ifstream in(data_path.data());
  if (in.is_open()) {
    std::cout << "Is Opened!\n";
  }
  std::vector<std::pair<double, double>> pairs;
  double x, y;
  while (in >> x >> y) {
    pairs.emplace_back(x, y);
  }
  in.close();
  return pairs;
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
  Variable<double> nn_input(input_shape), label(output_shape);

  auto pairs = GetInput("data.txt");
  optimizer.Apply(RandomNumberGenerator::NormalDistribution);

  for (int i = 0; i < 200; i++) {
    double iter_loss = 0;
    for (int j = 0; j < pairs.size(); j++) {
      auto& x_and_y = pairs[j];
      nn_input.values_.data_[0] = x_and_y.first;
      nn_input.grads_.data_[0] = 0;
      label.values_.data_[0] = x_and_y.second;
      label.grads_.data_[0] = 0;
      // std::cout << "nn_input:" << nn_input << std::endl;
      // std::cout << "label:" << label << std::endl;

      optimizer.ZeroGrad();
      auto* nn_output = mynn(&nn_input);
      // std::cout << "Output: " << nn_output << std::endl;
      auto* loss = LossFunc(nn_output, &label);
      // std::cout << "Loss:" << loss << std::endl;
      iter_loss += loss->values_.data_[0];
      loss->Backward();
      optimizer.Step();
    }
    std::cout << iter_loss << std::endl;
  }
  std::vector<std::pair<double, double>> x_and_ys;
  for (auto& [x, y] : pairs) {
    nn_input.values_.data_[0] = x;
    auto* nn_output = mynn(&nn_input);
    x_and_ys.emplace_back(x, nn_output->values_.data_[0]);
  }
  OutputResult(x_and_ys);
}
