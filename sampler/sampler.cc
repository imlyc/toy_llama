#include "sampler/sampler.h"

namespace tlm {

Sampler::Sampler() {}
Sampler::~Sampler() = default;

Token Sampler::Pick(std::span<const float> logits) {
  return Token::INVALID;
}

}  // namespace tlm
