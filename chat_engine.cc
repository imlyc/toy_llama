#include "chat_engine.h"

#include <glog/logging.h>

namespace tlm {

ChatEngine::ChatEngine(const Model& model) {}
ChatEngine::~ChatEngine() = default;

std::string ChatEngine::SendMessage(const std::string& message) {
  std::vector<Token> input_tokens = tokenizer_.TextToToken(message);
  std::vector<Token> output_tokens;

  transformer_.Prefill(input_tokens);

  Token next_token = NextToken(Token::BOS);
  while (next_token != Token::EOS) {
    output_tokens.push_back(next_token);
    next_token = NextToken(next_token);
  }

  return tokenizer_.TokenToText(output_tokens);
}

Token ChatEngine::NextToken(Token token) {
  if (token == Token::EOS) {
    return token;
  }

  std::vector<float> logits = transformer_.Predict(token);
  return sampler_.Pick(logits);
}

}  // namespace tlm
