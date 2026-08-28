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
  VectorView GetTokenEmbedding(Token token) const;

  Token GetBosToken() const;
  Token GetEosToken() const;

  int GetDecoderBlockCount() const;
  int GetVocabSize() const;

 private:
  VectorView GgufMatrixRow2VectorView(std::string_view key, int row) const;

  ScopedMmapFile model_file_;
  std::unique_ptr<GgufParser> parser_;
};

}  // namespace tlm
