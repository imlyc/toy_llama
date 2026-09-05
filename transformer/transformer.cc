#include "transformer/transformer.h"

#include <glog/logging.h>

#include "compute/storage.h"
#include "model/model.h"

namespace tlm {

Transformer::Transformer(const Model& model)
    : model_(model),
      token_embeddings_(model_.GetTokenEmbeddings()),
      final_rms_norm_(compute_engine_,
                      model_.GetOutputNormGamma(),
                      model_.GetLayerNormEpsilon()) {
  const int embedding_length = model_.GetTokenEmbeddingLength();
  embedding_storage_ = compute_engine_.Alloc(embedding_length);
  embedding_ = embedding_storage_->AsVector(embedding_length);

  const int decoder_block_count = model_.GetDecoderBlockCount();
  for (int i = 0; i < decoder_block_count; i++) {
    decoder_blocks_.emplace_back(compute_engine_);
  }

  const int64_t vocab_size = model_.GetVocabSize();
  logits_storage_ = compute_engine_.Alloc(vocab_size);
  logits_ = logits_storage_->AsVector(vocab_size);
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

  // Linear layer.
  compute_engine_.MatMul(logits_, token_embeddings_, final_rms_output);

  compute_engine_.Softmax(logits_);

  return logits_.As<const float>();
}

VectorView Transformer::LookupTokenEmbedding(Token token) {
  CHECK_NE(token.id, Token::INVALID.id);
  VectorView embedding = token_embeddings_.At(token.id);
  compute_engine_.Copy(embedding_, embedding);
  return embedding_.View();
}

}  // namespace tlm
