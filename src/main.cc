#include <iostream>
#include "op.h"
#include "tensor.h"
#include "variable.h"

int main() {
  TensorShape shape = {2, 2};
  Variable<double> va(shape), vb(shape);
  va.values_.data_[0] = -0.6727;
  va.values_.data_[1] = 1.2163;
  va.values_.data_[2] = 1.9147;
  va.values_.data_[3] = -1.6147;
  vb.values_.data_[0] = -0.4696;
  vb.values_.data_[1] = 0.5412;
  vb.values_.data_[2] = 0.2462;
  vb.values_.data_[3] = 1.0296;
  auto vc = MM<double>(va, vb);
  auto mean = Mean<double>(vc);
  Backward<double>(mean);
  std::cout << vc << '\n' << mean << std::endl;
  std::cout << "==================" << std::endl;
  std::cout << va << '\n' << vb << std::endl;
}
