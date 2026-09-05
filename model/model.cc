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

MatrixView Model::GetTokenEmbeddings() const {
  return GgufMatrix2MatrixView("token_embd.weight");
}

int Model::GetTokenEmbeddingLength() const {
  return CheckedCast<int>(
      parser_->GetMetadata<uint32_t>("llama.embedding_length"));
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

int Model::GetVocabSize() const {
  return CheckedCast<int>(parser_->GetMetadata<uint32_t>("llama.vocab_size"));
}

VectorView Model::GetOutputNormGamma() const {
  return GgufVector2VectorView("output_norm.weight");
}

float Model::GetLayerNormEpsilon() const {
  return parser_->GetMetadata<float>("llama.attention.layer_norm_rms_epsilon");
}

MatrixView Model::GgufMatrix2MatrixView(std::string_view key) const {
  const GgufParser::TensorInfo& info = parser_->GetTensorInfo(key);
  const std::byte* data = parser_->GetTensorData(info);

  DType dtype = GgmlType2DType(info.type);
  CHECK_EQ(info.dimensions.size(), 2);
  return MatrixView::Create(dtype, data,
                            CheckedCast<int64_t>(info.dimensions[1]),
                            CheckedCast<int64_t>(info.dimensions[0]));
}

VectorView Model::GgufVector2VectorView(std::string_view key) const {
  const GgufParser::TensorInfo& info = parser_->GetTensorInfo(key);
  const std::byte* data = parser_->GetTensorData(info);

  DType dtype = GgmlType2DType(info.type);
  CHECK_EQ(info.dimensions.size(), 1);
  return VectorView::Create(dtype, data,
                            CheckedCast<int64_t>(info.dimensions[0]));
}
}  // namespace tlm
