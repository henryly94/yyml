#ifndef NN_H
#define NN_H

#include <unordered_map>
#include <utility>
#include "dense_layer.h"
#include "variable.h"

class NN {
 public:
  virtual Variable<double>* Forward(Variable<double>*) = 0;

  Variable<double>* operator()(Variable<double>* input) {
    return Forward(input);
  }

  void print() {
    for (auto it : layers_) {
      it.second->print();
    }
  }

 protected:
  template <typename... Params>
  void set_layer(std::string name, Params&... params) {
    auto* new_layer =
        DenseLayer::factory::GetNewInstance(std::forward<Params>(params)...);
    layers_.emplace(name, new_layer);
  }

  LayerInterface* get_layer(std::string name) {
    LayerInterface* ret = nullptr;
    auto pair = layers_.find(name);
    if (pair != layers_.end()) {
      ret = pair->second;
    }
    return ret;
  }

 private:
  std::unordered_map<std::string, LayerInterface*> layers_;
};

#endif  // NN_H
