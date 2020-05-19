#ifndef FACTORY_H
#define FACTORY_H

#include <functional>
#include <memory>
#include <set>
#include <unordered_set>
#include <utility>

namespace yyml {

namespace detail {

template <typename Type>
struct PointerComp {
  using is_transparent = void;

  struct helper {
    helper() : ptr(nullptr) {}
    helper(helper const &) = default;
    helper(Type *ptr) : ptr(ptr) {}
    helper(std::unique_ptr<Type> const &unique_ptr) : ptr(unique_ptr.get()) {}

    bool operator<(helper o) const { return std::less<Type *>()(ptr, o.ptr); }

    Type *ptr;
  };

  bool operator()(helper const &&lhs, helper const &&rhs) const {
    return lhs < rhs;
  }
};

}  // namespace detail

template <typename Type>
class Factory {
 public:
  using register_type =
      std::set<std::unique_ptr<Type>, detail::PointerComp<Type>>;

  template <typename... Params>
  static Type *GetNewInstance(Params &&... params) {
    auto t = std::make_unique<Type>(std::forward<Params>(params)...);
    auto *ret = t.get();
    registry().insert(std::move(t));
    return ret;
  }

  static void RemoveInstance(Type *instance) {
    if (instance != nullptr && registry().find(instance) != registry().end()) {
      auto iter = registry().find(instance);
      registry().erase(iter);
    }
  }

 private:
  static register_type &registry() {
    static register_type _;
    return _;
  }
};

}  // namespace yyml

#endif  // FACTORY_H
