#include "chat_engine.h"

#include <glog/logging.h>

namespace tlm {

ChatEngine::ChatEngine(const Model& model)
    : model_(model),
      tokenizer_(model),
      chat_template_(tokenizer_),
      transformer_(model) {}
ChatEngine::~ChatEngine() = default;

void ChatEngine::SendMessageHelper(const std::string& message,
                                   TokenCb cb) {
  std::vector<Token> message_tokens = tokenizer_.TextToToken(message);

  // Insert BOS on first message. Also insert EOS on other messages to make sure
  // the EOS token goes into KV cache for multi turn chat.
  std::vector<Token> input_tokens = chat_template_.Apply(
      message_tokens,
      first_message_received_ ? model_.GetEosToken() : model_.GetBosToken());
  first_message_received_ = true;

  std::span<const float> logits = transformer_.Prefill(input_tokens);

  Token next_token = sampler_.Pick(logits);
  while (next_token != model_.GetEosToken() && next_token != Token::INVALID) {
    cb(next_token);
    logits = transformer_.Predict(next_token);
    next_token = sampler_.Pick(logits);
  }
}

void ChatEngine::SendMessageAsync(const std::string& message,
                                  SendMessageCb cb) {
  SendMessageHelper(message, [this, cb](Token token) {
    cb(tokenizer_.TokenToText({token}));
  });
}

std::string ChatEngine::SendMessage(const std::string& message) {
  std::vector<Token> output;
  SendMessageHelper(message, [&output](Token token) {
    output.push_back(token);
  });

  return tokenizer_.TokenToText(output);
}

}  // namespace tlm
