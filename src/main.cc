#include <iostream>
#include "op.h"
#include "tensor.h"
#include "variable.h"

int main() {
  TensorShape shape = {2, 3};
  Variable<double> va(shape), vb(shape), vc(shape);
  auto va_b = Add<double>(va, vb);
  auto vab_c = Substract<double>(va_b, vc);
  auto va_abc = Add<double>(va, vab_c);
  auto va_a_abc = Substract<double>(va_abc, va);
  std::cout << va << '\n' << vb << '\n' << vc << std::endl;
  Backward<double>(va_a_abc);
  std::cout << va << '\n' << vb << '\n' << vc << std::endl;
}
