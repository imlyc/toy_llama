#include "transformer.h"

namespace tlm {

Transformer::Transformer() {}
Transformer::~Transformer() = default;

std::vector<float> Transformer::Prefill(const std::vector<Token>& tokens) {
  return {};
}

std::vector<float> Transformer::Predict(Token token) {
  return {};
}

}  // namespace tlm
