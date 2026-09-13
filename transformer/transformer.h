#pragma once

#include <span>
#include <vector>

#include "compute/compute_engine.h"
#include "model/token.h"
#include "tensor/tensor_view.h"
#include "transformer/decoder_block.h"
#include "transformer/rms_norm_layer.h"
#include "transformer/tensor_storage.h"

namespace tlm {
class Model;
class Storage;

class Transformer {
 public:
  explicit Transformer(const Model& model);
  ~Transformer();

  std::span<const float> Prefill(std::span<const Token> tokens);
  std::span<const float> Predict(Token token);

 private:
  MatrixView LookupTokenEmbeddings(std::span<const Token> tokens);

  const Model& model_;
  ComputeEngine compute_engine_;

  const int embedding_size_;
  MatrixView token_embeddings_;

  MatrixStorage embeddings_;
  VectorStorage logits_;

  std::vector<DecoderBlock> decoder_blocks_;
  RmsNormLayer final_rms_norm_;
};

}  // namespace tlm
