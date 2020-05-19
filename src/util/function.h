#ifndef FUNCTION_H
#define FUNCTION_H

#include <random>

namespace yyml {

class RandomNumberGenerator {
 public:
  static double NormalDistribution() { return normal_(generator_); }

 private:
  static std::default_random_engine generator_;
  static std::normal_distribution<double> normal_;
};

}  // namespace yyml

#endif  // FUNCTION_H
