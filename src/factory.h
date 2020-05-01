#ifndef FACTORY_H
#define FACTORY_H

#include <memory>
#include <unordered_set>
#include <utility>

template <typename Type>
class Factory {
 public:
  using register_type = std::unordered_set<Type *>;

  template <typename... Params>
  static Type *GetNewInstance(Params &&... params) {
    Type *t = new Type(std::forward<Params>(params)...);
    registry().insert(t);
    return t;
  }

  static void RemoveInstance(Type *instance) {
    if (instance != nullptr && registry().find(instance) != registry().end()) {
      registry().erase(instance);
      delete instance;
    }
  }

 private:
  static register_type &registry() {
    static register_type _;
    return _;
  }
};

#endif  // FACTORY_H
