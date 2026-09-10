#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include "base/mmap_file.h"
#include "model/token.h"
#include "tensor/tensor_view.h"

namespace tlm {
class GgufParser;

class Model {
 public:
  Model(ScopedMmapFile file, std::unique_ptr<GgufParser> parser);
  ~Model();

  const std::vector<std::string_view>& GetTokenizerMerges() const;
  const std::vector<std::string_view>& GetTokenizerTokens() const;
  MatrixView GetTokenEmbeddings() const;
  int GetTokenEmbeddingLength() const;

  Token GetBosToken() const;
  Token GetEosToken() const;

  int GetDecoderBlockCount() const;
  int GetContextLength() const;
  int GetEmbeddingLength() const;
  int GetVocabSize() const;

  VectorView GetOutputNormGamma() const;
  float GetLayerNormEpsilon() const;

  int GetAttnKeyLength() const;
  int GetAttnValueLength() const;
  int GetAttnHeadCount() const;
  int GetAttnHeadCountKv() const;

  float GetRopeFreqBase() const;
  int GetRopeDimensionCount() const;
  VectorView GetRopeFreqsWeight() const;

  MatrixView GetDecoderBlockAttnQWeight(int index) const;
  MatrixView GetDecoderBlockAttnKWeight(int index) const;
  MatrixView GetDecoderBlockAttnVWeight(int index) const;
  MatrixView GetDecoderBlockAttnOWeight(int index) const;

  MatrixView GetDecoderBlockFfnUpWeight(int index) const;
  MatrixView GetDecoderBlockFfnGateWeight(int index) const;
  MatrixView GetDecoderBlockFfnDownWeight(int index) const;

  VectorView GetDecoderBlockAttnNormWeight(int index) const;
  VectorView GetDecoderBlockFfnNormWeight(int index) const;

  std::string_view GetChatTemplate() const;

 private:
  MatrixView GgufMatrix2MatrixView(std::string_view key) const;
  VectorView GgufVector2VectorView(std::string_view key) const;

  ScopedMmapFile model_file_;
  std::unique_ptr<GgufParser> parser_;
};

}  // namespace tlm
