#pragma once

#include <string>
#include <vector>

#include "token.h"

namespace tlm {

class Tokenizer {
 public:
  Tokenizer();
  ~Tokenizer();

  std::vector<Token> TextToToken(const std::string& text);
  std::string TokenToText(const std::vector<Token>& tokens);
};

}  // namespace tlm
