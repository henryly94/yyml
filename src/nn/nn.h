#ifndef NN_H
#define NN_H

#include <string_view>
#include <unordered_map>
#include <utility>
#include "nn/layer/dense_layer.h"
#include "nn/variable.h"

namespace yyml {
namespace nn {

class NN {
 public:
  virtual Variable<double>* Forward(Variable<double>*) = 0;

  Variable<double>* operator()(Variable<double>* input) {
    return Forward(input);
  }

  std::vector<Variable<double>*> Parameters() {
    std::vector<Variable<double>*> parameters;
    for (auto name_and_layer_pair : layers_) {
      auto* layer = name_and_layer_pair.second;
      auto layer_params = layer->Parameters();
      parameters.insert(parameters.end(), layer_params.begin(),
                        layer_params.end());
    }
    return parameters;
  }

  void print() {
    for (auto it : layers_) {
      it.second->print();
    }
  }

 protected:
  template <typename... Params>
  void SetLayer(std::string_view name, Params&... params) {
    auto* new_layer = DenseLayer::factory::GetNewInstance(
        std::forward<Params>(params)..., std::string(name));
    layers_.emplace(std::string(name), new_layer);
  }

  LayerInterface& GetLayer(std::string_view name) {
    LayerInterface* ret = nullptr;
    auto pair = layers_.find(std::string(name));
    if (pair != layers_.end()) {
      ret = pair->second;
    }
    return *ret;
  }

 private:
  std::unordered_map<std::string, LayerInterface*> layers_;
};

}  // namespace nn
}  // namespace yyml

#endif  // NN_H
