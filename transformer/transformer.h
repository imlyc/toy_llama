#pragma once

#include <span>
#include <vector>

#include "model/token.h"
#include "tensor/tensor_view.h"
#include "transformer/decoder_block.h"
#include "transformer/linear_layer.h"
#include "transformer/rms_norm_layer.h"

namespace tlm {
class Model;

class Transformer {
 public:
  explicit Transformer(const Model& model);
  ~Transformer();

  std::span<const float> Prefill(std::span<const Token> tokens);
  std::span<const float> Predict(Token token);

 private:
  VectorView LookupTokenEmbedding(Token token);

  const Model& model_;
  std::vector<DecoderBlock> decoder_blocks_;
  RmsNormLayer final_rms_norm_;
  LinearLayer linear_output_;
};

}  // namespace tlm
