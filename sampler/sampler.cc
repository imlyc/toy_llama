#include "sampler/sampler.h"

#include <algorithm>

#include "base/cast.h"

namespace tlm {

Sampler::Sampler() {}
Sampler::~Sampler() = default;

Token Sampler::Pick(std::span<const float> logits) {
  auto largest = std::max_element(logits.begin(), logits.end());
  return {.id = CheckedCast<int32_t>(largest - logits.begin())};
}

}  // namespace tlm
