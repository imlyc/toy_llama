#pragma once

#include <string>
#include <functional>

#include "model/model.h"
#include "model/token.h"
#include "sampler/sampler.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/tokenizer.h"
#include "transformer/transformer.h"

namespace tlm {

class ChatEngine {
 public:
  using SendMessageCb = std::function<void(const std::string&)>;

  explicit ChatEngine(const Model& model);
  ~ChatEngine();

  std::string SendMessage(const std::string& message);
  void SendMessageAsync(const std::string& message, SendMessageCb cb);

 private:
  using TokenCb = std::function<void(Token)>;
  void SendMessageHelper(const std::string& message, TokenCb cb);

  const Model& model_;
  Tokenizer tokenizer_;
  ChatTemplate chat_template_;
  Transformer transformer_;
  Sampler sampler_;

  bool first_message_received_ = false;
};

}  // namespace tlm
