#pragma once

#include <span>
#include <vector>

#include "compute/compute_engine.h"
#include "model/token.h"
#include "tensor/tensor_view.h"
#include "transformer/decoder_block.h"
#include "transformer/rms_norm_layer.h"

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
  VectorView LookupTokenEmbedding(Token token);

  const Model& model_;
  ComputeEngine compute_engine_;

  MatrixView token_embeddings_;

  std::unique_ptr<Storage> embedding_storage_;
  MutableVectorView embedding_;

  std::vector<DecoderBlock> decoder_blocks_;
  RmsNormLayer final_rms_norm_;

  std::unique_ptr<Storage> logits_storage_;
  MutableVectorView logits_;
};

}  // namespace tlm
