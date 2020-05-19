#include <fstream>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>
#include "base/tensor.h"
#include "nn/layer/dense_layer.h"
#include "nn/nn.h"
#include "nn/op/op.h"
#include "nn/optimizer/sgd_optimizer.h"
#include "nn/variable.h"
#include "util/data_loader.h"
#include "util/function.h"

using yyml::TensorShape;
using yyml::nn::Conv2D;
using yyml::nn::Variable;

int main() {
  TensorShape input_shape{3, 3}, kernel_shape{2, 2};
  Variable<double> input(input_shape), kernel(kernel_shape);

  for (size_t i = 0; i < 9; i++) {
    input.values_.data_[i] = i;
  }
  for (size_t j = 0; j < 4; j++) {
    kernel.values_.data_[j] = j;
  }

  auto* result = Conv2D<double>(&input, &kernel);
  std::cout << result << std::endl;
  result->Backward();

  std::cout << input << std::endl;
  std::cout << kernel << std::endl;
}
