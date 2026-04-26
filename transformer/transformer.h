#pragma once

#include <vector>

#include "model/token.h"

namespace tlm {

class Transformer {
 public:
  Transformer();
  ~Transformer();

  std::vector<float> Prefill(const std::vector<Token>& tokens);
  std::vector<float> Predict(Token token);
};

}  // namespace tlm
