#include "function.h"

namespace yyml {

std::default_random_engine RandomNumberGenerator::generator_;

std::normal_distribution<double> RandomNumberGenerator::normal_(0, 0.1);

}  // namespace yyml
