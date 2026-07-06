#pragma once

#include <span>

#include "model/token.h"

namespace tlm {

class Sampler {
 public:
  Sampler();
  ~Sampler();

  Token Pick(std::span<const float> logits);
};

}  // namespace tlm
