#include <iostream>
#include "dense_layer.h"
#include "op.h"
#include "tensor.h"
#include "variable.h"

// class MyNN : public NN {
// public:
//  MyNN() {
//    dense_1 = new DenseLayer({20, 256});
//    dense_2 = new DenseLayer({256, 16});
//  }
//
//  ~MyNN() {
//    delete dense_1;
//    delete dense_2;
//  }
//
//  Variable<double> Forward(Variable<double>& input) override {
//    auto o1 = dense_1(input);
//    auto a1 = ReLU<double>(o1);
//    auto o2 = dense_2(a1);
//    auto a2 = ReLU<double>(o2);
//    return a2;
//  }
//
// private:
//  DenseLayer* dense_1;
//  DenseLayer* dense_2;
//};

int main() {
  TensorShape shape = {2, 2};

  Variable<double> va(shape), vb(shape), bias(shape);
  va.values_.data_[0] = 1;
  va.values_.data_[1] = 2;
  va.values_.data_[2] = 3;
  va.values_.data_[3] = 4;
  vb.values_.data_[0] = 4;
  vb.values_.data_[1] = 3;
  vb.values_.data_[2] = 2;
  vb.values_.data_[3] = 1;
  bias.values_.data_[0] = bias.values_.data_[1] = bias.values_.data_[2] =
      bias.values_.data_[3] = 1;
  auto* vc = MM<double>(&va, &vb);
  auto* vd = Add<double>(vc, &bias);
  // auto* mean = Mean<double>(vd);
  std::cout << vc << '\n' << vd << std::endl;
  Backward<double>(vd);
  std::cout << "==================" << std::endl;
  std::cout << va << '\n' << vb << '\n' << bias << std::endl;

  DenseLayer dense(shape, shape);
  Variable<double> input(shape);
  input.values_.data_[0] = 1;
  input.values_.data_[1] = 2;
  input.values_.data_[2] = 3;
  input.values_.data_[3] = 4;
  dense.weight_.values_.data_[0] = 4;
  dense.weight_.values_.data_[1] = 3;
  dense.weight_.values_.data_[2] = 2;
  dense.weight_.values_.data_[3] = 1;
  dense.bias_.values_.data_[0] = dense.bias_.values_.data_[1] =
      dense.bias_.values_.data_[2] = dense.bias_.values_.data_[3] = 1;
  auto* output = dense(&input);
  std::cout << output << std::endl;
  Backward<double>(output);
  std::cout << input << '\n'
            << dense.weight_ << '\n'
            << dense.bias_ << std::endl;
}
