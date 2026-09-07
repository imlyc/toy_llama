#include "model/model.h"

#include <sstream>
#include <string>
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

std::string DecoderBlockTensorName(int index, std::string_view name) {
  std::stringstream ss;
  ss << "blk." << index << "." << name;
  return ss.str();
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

int Model::GetContextLength() const {
  return CheckedCast<int>(
      parser_->GetMetadata<uint32_t>("llama.context_length"));
}

int Model::GetEmbeddingLength() const {
  return CheckedCast<int>(
      parser_->GetMetadata<uint32_t>("llama.embedding_length"));
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

int Model::GetAttnKeyLength() const {
  return CheckedCast<int>(
      parser_->GetMetadata<uint32_t>("llama.attention.key_length"));
}

int Model::GetAttnValueLength() const {
  return CheckedCast<int>(
      parser_->GetMetadata<uint32_t>("llama.attention.value_length"));
}

int Model::GetAttnHeadCount() const {
  return CheckedCast<int>(
      parser_->GetMetadata<uint32_t>("llama.attention.head_count"));
}

int Model::GetAttnHeadCountKv() const {
  return CheckedCast<int>(
      parser_->GetMetadata<uint32_t>("llama.attention.head_count_kv"));
}

float Model::GetRopeFreqBase() const {
  return parser_->GetMetadata<float>("llama.rope.freq_base");
}

int Model::GetRopeDimensionCount() const {
  return CheckedCast<int>(
      parser_->GetMetadata<uint32_t>("llama.rope.dimension_count"));
}

VectorView Model::GetRopeFreqsWeight() const {
  return GgufVector2VectorView("rope_freqs.weight");
}

MatrixView Model::GetDecoderBlockAttnQWeight(int index) const {
  return GgufMatrix2MatrixView(DecoderBlockTensorName(index, "attn_q.weight"));
}

MatrixView Model::GetDecoderBlockAttnKWeight(int index) const {
  return GgufMatrix2MatrixView(DecoderBlockTensorName(index, "attn_k.weight"));
}

MatrixView Model::GetDecoderBlockAttnVWeight(int index) const {
  return GgufMatrix2MatrixView(DecoderBlockTensorName(index, "attn_v.weight"));
}

MatrixView Model::GetDecoderBlockAttnOWeight(int index) const {
  return GgufMatrix2MatrixView(
      DecoderBlockTensorName(index, "attn_output.weight"));
}

MatrixView Model::GetDecoderBlockFfnUpWeight(int index) const {
  return GgufMatrix2MatrixView(DecoderBlockTensorName(index, "ffn_up.weight"));
}

MatrixView Model::GetDecoderBlockFfnGateWeight(int index) const {
  return GgufMatrix2MatrixView(
      DecoderBlockTensorName(index, "ffn_gate.weight"));
}

MatrixView Model::GetDecoderBlockFfnDownWeight(int index) const {
  return GgufMatrix2MatrixView(
      DecoderBlockTensorName(index, "ffn_down.weight"));
}

VectorView Model::GetDecoderBlockAttnNormWeight(int index) const {
  return GgufVector2VectorView(
      DecoderBlockTensorName(index, "attn_norm.weight"));
}

VectorView Model::GetDecoderBlockFfnNormWeight(int index) const {
  return GgufVector2VectorView(
      DecoderBlockTensorName(index, "ffn_norm.weight"));
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
