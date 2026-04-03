#pragma once

#include <vector>

#include "token.h"

namespace tlm {

class Sampler {
 public:
  Sampler();
  ~Sampler();

  Token Pick(const std::vector<float>& logits);
};

}  // namespace tlm
