#pragma once

#include <vector>

#include "model/token.h"

namespace tlm {

class Sampler {
 public:
  Sampler();
  ~Sampler();

  Token Pick(const std::vector<float>& logits);
};

}  // namespace tlm
