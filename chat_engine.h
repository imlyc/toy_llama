#pragma once

#include <string>

#include "model.h"

namespace tlm {

class ChatEngine {
 public:
  explicit ChatEngine(const Model& model);
  ~ChatEngine();

  std::string SendMessage(const std::string& message);

 private:
  Token NextToken(Token token);

  Tokenizer tokenizer_;
  Transformer transfomer_;
  Sampler sampler_;
};

}  // namespace tlm
