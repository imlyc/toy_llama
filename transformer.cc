#include "transformer.h"

namespace tlm {

Transformer::Transformer() {}
Transformer::~Transformer() = default;

void Transformer::Prefill(const std::vector<Token>& tokens) {}

std::vector<float> Transformer::Predict(Token token) {
  return {};
}

}  // namespace tlm
