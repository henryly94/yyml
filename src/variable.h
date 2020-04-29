#ifndef VARIABLE_H
#define VARIABLE_H

#include <functional>
#include <initializer_list>
#include <numeric>
#include <vector>

struct VariableShape {
  VariableShape(std::initializer_list<size_t> dims)
      : dim(dims.size()),
        dims(dims),
        total(std::accumulate(dims.begin(), dims.end(), 1,
                              std::multiplies<size_t>())) {}
  size_t dim;
  size_t total;
  std::vector<size_t> dims;
};

template <typename Type>
struct Variable {
  Variable(VariableShape shape) : shape(shape), data(new Type[shape.total]) {}
  ~Variable() { delete[] data; }
  Type* data;
  VariableShape shape;
};

using DoubleVariable = Variable<double>;

#endif  // VARIABLE_H
