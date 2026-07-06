#include "transformer/transformer.h"

#include <glog/logging.h>

#include "model/model.h"

namespace tlm {

Transformer::Transformer(const Model& model) : model_(model) {
  int decoder_block_count = model_.GetDecoderBlockCount();
  for (int i = 0; i < decoder_block_count; i++) {
    decoder_blocks_.emplace_back();
  }
}

Transformer::~Transformer() = default;

std::span<const float> Transformer::Prefill(std::span<const Token> tokens) {
  // A very simple version of Prefill is to call Predict on each token.
  std::span<const float> logits;
  for (const auto& token : tokens) {
    logits = Predict(token);
  }
  return logits;
}

std::span<const float> Transformer::Predict(Token token) {
  VectorView block_input = LookupTokenEmbedding(token);
  for (auto& decoder_block : decoder_blocks_) {
    block_input = decoder_block.Forward(block_input);
  }

  VectorView final_rms_output = final_rms_norm_.Forward(block_input);
  VectorView linear_output = linear_output_.Forward(final_rms_output);

  CHECK_EQ(linear_output.dtype, DType::F32);
  return linear_output.As<const float>();
}

VectorView Transformer::LookupTokenEmbedding(Token token) {
  return model_.GetTokenEmbedding(token);
}

}  // namespace tlm
