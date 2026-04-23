#include "model.h"

#include <string_view>

#include <glog/logging.h>

#include "gguf/gguf_parser.h"

namespace tlm {

Model::Model(ScopedMmapFile file, std::unique_ptr<GgufParser> parser)
    : model_file_(std::move(file)),
      parser_(std::move(parser)) {}

Model::~Model() = default;

const std::vector<std::string_view>& Model::GetTokenizerMerges() const {
  const GgufParser::Array& merges =
      parser_->GetMetadata<GgufParser::Array>("tokenizer.ggml.merges");
  CHECK(!merges.strings.empty());
  return merges.strings;
}

const std::vector<std::string_view>& Model::GetTokenizerTokens() const {
  return parser_->GetMetadata<std::vector<std::string_view>>(
      "tokenizer.ggml.tokens");
}
}  // namespace tlm
