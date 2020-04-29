#include <iostream>
#include "variable.h"

int main() {
  VariableShape a = {1, 2, 3};
  VariableShape b{4, 5};
  std::cout << a.total << ' ' << a.dim << ' ' << b.total << ' ' << b.dim
            << std::endl;

  Variable<double> v(b);
  v.data[10] = 4;
}
