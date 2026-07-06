#include "chat_engine.h"

#include <glog/logging.h>

namespace tlm {

ChatEngine::ChatEngine(const Model& model)
    : model_(model), tokenizer_(model), transformer_(model) {}
ChatEngine::~ChatEngine() = default;

std::string ChatEngine::SendMessage(const std::string& message) {
  std::vector<Token> input_tokens = tokenizer_.TextToToken(message);
  std::vector<Token> output_tokens;

  std::span<const float> logits = transformer_.Prefill(input_tokens);

  Token next_token = sampler_.Pick(logits);
  while (next_token != model_.GetEosToken() && next_token != Token::INVALID) {
    output_tokens.push_back(next_token);

    logits = transformer_.Predict(next_token);
    next_token = sampler_.Pick(logits);
  }

  return tokenizer_.TokenToText(output_tokens);
}

}  // namespace tlm
