#pragma once

#include <string>

#include "model/model.h"
#include "model/token.h"
#include "sampler/sampler.h"
#include "tokenizer/tokenizer.h"
#include "transformer/transformer.h"

namespace tlm {

class ChatEngine {
 public:
  explicit ChatEngine(const Model& model);
  ~ChatEngine();

  std::string SendMessage(const std::string& message);

 private:
  Tokenizer tokenizer_;
  Transformer transformer_;
  Sampler sampler_;
};

}  // namespace tlm
