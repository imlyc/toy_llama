#include "sampler.h"

namespace tlm {

Sampler::Sampler() {}
Sampler::~Sampler() = default;

Token Sampler::Pick(const std::vector<float>& logits) {
  return Token::EOS;
}

}  // namespace tlm
