#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <fstream>
#include <iterator>
#include <sstream>
#include <string_view>
#include <tuple>
#include <utility>
#include "tensor.h"
#include "variable.h"

class DataLoader {
 public:
  using container_type =
      std::vector<std::pair<Variable<double>, Variable<double>>>;
  using iterator_type = container_type::iterator;

  DataLoader(std::string_view path, TensorShape input_shape,
             TensorShape label_shape) {
    std::ifstream in(path.data());
    if (!in.is_open()) {
      std::cerr << "IOError: when opening " << path << std::endl;
      return;
    }

    // Read line first since eof() won't work unless get an extra read.
    std::string line;
    std::getline(in, line, '\n');
    while (!in.eof()) {
      std::istringstream iss(line);
      auto iter = data_.emplace(data_.end(), std::piecewise_construct,
                                std::forward_as_tuple(input_shape),
                                std::forward_as_tuple(label_shape));
      auto& input_variable = iter->first;
      auto& label_variable = iter->second;
      for (size_t i = 0; i < input_shape.total; i++) {
        iss >> input_variable.values_.data_[i];
      }

      for (size_t j = 0; j < label_shape.total; j++) {
        iss >> label_variable.values_.data_[j];
      }
      std::getline(in, line, '\n');
    }
    in.close();
  }

  iterator_type begin() { return data_.begin(); }
  iterator_type end() { return data_.end(); }

  size_t size() { return data_.size(); }

  container_type data_;
};

#endif  // DATA_LOADER_H
