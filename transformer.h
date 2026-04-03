#pragma once

#include <vector>

#include "token.h"

namespace tlm {

class Transformer {
 public:
  Transformer();
  ~Transformer();

  void Prefill(const std::vector<Token>& tokens);
  std::vector<float> Predict(Token token);
};

}  // namespace tlm
