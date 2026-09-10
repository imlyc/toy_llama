#include "tokenizer/chat_template.h"

#include <glog/logging.h>

#include "tokenizer/tokenizer.h"

namespace tlm {
namespace {
inline void Append(std::vector<Token>& prompt, Token token) {
  prompt.push_back(token);
}

inline void Append(std::vector<Token>& prompt,
                   const std::vector<Token> tokens) {
  prompt.insert(prompt.end(), tokens.begin(), tokens.end());
}
}  // namespace

ChatTemplate::ChatTemplate(Tokenizer& tokenizer)
  : start_header_id_(tokenizer.GetToken("<|start_header_id|>")),
    end_header_id_(tokenizer.GetToken("<|end_header_id|>")),
    eot_id_(tokenizer.GetToken("<|eot_id|>")),
    user_(tokenizer.TextToToken("user")),
    assistant_(tokenizer.TextToToken("assistant")),
    newline_(tokenizer.TextToToken("\n\n")) {
  CHECK_NE(start_header_id_, Token::INVALID);
  CHECK_NE(end_header_id_, Token::INVALID);
  CHECK_NE(eot_id_, Token::INVALID);
  CHECK(!user_.empty());
  CHECK(!assistant_.empty());
  CHECK(!newline_.empty());
}
ChatTemplate::~ChatTemplate() = default;

std::vector<Token> ChatTemplate::Apply(const std::vector<Token>& message,
                                       Token insert_token) {
  std::vector<Token> prompt;
  Append(prompt, insert_token);
  Append(prompt, start_header_id_);
  Append(prompt, user_);
  Append(prompt, end_header_id_);
  Append(prompt, newline_);
  Append(prompt, message);
  Append(prompt, eot_id_);
  Append(prompt, start_header_id_);
  Append(prompt, assistant_);
  Append(prompt, end_header_id_);
  Append(prompt, newline_);
  return prompt;
}

}  // namespace tlm
