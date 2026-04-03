#include "tokenizer.h"

namespace tlm {

Tokenizer::Tokenizer() {}
Tokenizer::~Tokenizer() = default;

std::vector<Token> Tokenizer::TextToToken(const std::string& text) {
  return {};
}

std::string Tokenizer::TokenToText(const std::vector<Token>& tokens) {
  return "";
}

}  // namespace tlm
