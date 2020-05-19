#ifndef LAYER_INTERFACE_H
#define LAYER_INTERFACE_H

#include <vector>
#include "nn/variable.h"

namespace yyml {
namespace nn {

class LayerInterface {
 public:
  virtual Variable<double>* operator()(Variable<double>* input) = 0;

  virtual std::vector<Variable<double>*> Parameters() = 0;

  virtual void print() = 0;
};

}  // namespace nn
}  // namespace yyml

#endif  // LAYER_INTERFACE_H
