#pragma once

#include <vector>

#include "model/token.h"

namespace tlm {
class Tokenizer;

// Chat template for instruct tuned models. This impl is for llama3.2.
// TODO: Create template using tokenizer.chat_template.
class ChatTemplate {
 public:
  explicit ChatTemplate(Tokenizer& tokenizer);
  ~ChatTemplate();

  // Apply the chat template on `message`. Insert `insert_token` at the
  // beginning of the message.
  std::vector<Token> Apply(const std::vector<Token>& message,
                           Token insert_token);
 private:
  const Token start_header_id_;
  const Token end_header_id_;
  const Token eot_id_;
  const std::vector<Token> user_;
  const std::vector<Token> assistant_;
  const std::vector<Token> newline_; 
};

}  // namespace tlm
