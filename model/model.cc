#include "model/model.h"

#include <string_view>

#include <glog/logging.h>

#include "base/cast.h"
#include "gguf/gguf_parser.h"
#include "tensor/dtype.h"

namespace tlm {
namespace {
DType GgmlType2DType(uint32_t type) {
  switch (static_cast<GgmlType>(type)) {
    case GgmlType::F32:
      return DType::F32;
    case GgmlType::Q8_0:
      return DType::Q8_0;
    default:
      LOG(FATAL) << "Unsupported ggml type " << type;
      return DType::F32;
  }
}
}  // namespace

Model::Model(ScopedMmapFile file, std::unique_ptr<GgufParser> parser)
    : model_file_(std::move(file)), parser_(std::move(parser)) {}

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

VectorView Model::GetTokenEmbedding(Token token) const {
  CHECK_NE(token.id, Token::INVALID.id);
  CHECK_GE(token.id, 0);
  return GgufMatrixRow2VectorView("token_embd.weight", token.id);
}

Token Model::GetBosToken() const {
  uint32_t token_id =
      parser_->GetMetadata<uint32_t>("tokenizer.ggml.bos_token_id");
  return Token{.id = CheckedCast<int32_t>(token_id)};
}

Token Model::GetEosToken() const {
  uint32_t token_id =
      parser_->GetMetadata<uint32_t>("tokenizer.ggml.eos_token_id");
  return Token{.id = CheckedCast<int32_t>(token_id)};
}

int Model::GetDecoderBlockCount() const {
  return CheckedCast<int>(parser_->GetMetadata<uint32_t>("llama.block_count"));
}

VectorView Model::GgufMatrixRow2VectorView(std::string_view key,
                                           int row) const {
  const GgufParser::TensorInfo& info = parser_->GetTensorInfo(key);
  const std::byte* data = parser_->GetTensorData(info);
  CHECK_LT(row, info.dimensions[1]);

  DType dtype = GgmlType2DType(info.type);

  VectorView vv{
      .dtype = dtype,
      .data = data + info.dimensions[0] * row,
      .shape = {CheckedCast<int>(info.dimensions[0])},
      .stride = {GetDTypeBlockBytes(dtype)},
  };

  return vv;
}
}  // namespace tlm
