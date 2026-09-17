#include "transformer/transformer.h"

#include <algorithm>

#include <glog/logging.h>

#include "compute/storage.h"
#include "model/model.h"

namespace tlm {
namespace {
DecoderBlock::Param GetDecoderBlockParam(const Model& model, int index) {
  float epsilon = model.GetLayerNormEpsilon();

  DecoderBlock::Param param;

  param.embedding_size = model.GetEmbeddingLength();

  param.attn_norm = {model.GetDecoderBlockAttnNormWeight(index), epsilon};

  param.gqa.key_size = model.GetAttnKeyLength();
  param.gqa.value_size = model.GetAttnValueLength();
  param.gqa.context_length = model.GetContextLength();
  param.gqa.head_count = model.GetAttnHeadCount();
  param.gqa.head_count_kv = model.GetAttnHeadCountKv();
  param.gqa.rope_freq_base = model.GetRopeFreqBase();
  param.gqa.rope_dimension_count = model.GetRopeDimensionCount();
  param.gqa.wq = model.GetDecoderBlockAttnQWeight(index);
  param.gqa.wk = model.GetDecoderBlockAttnKWeight(index);
  param.gqa.wv = model.GetDecoderBlockAttnVWeight(index);
  param.gqa.wo = model.GetDecoderBlockAttnOWeight(index);
  param.gqa.rope_freqs = model.GetRopeFreqsWeight();

  param.ffn_norm = {model.GetDecoderBlockFfnNormWeight(index), epsilon};

  param.swiglu_ffn.wup = model.GetDecoderBlockFfnUpWeight(index);
  param.swiglu_ffn.wgate = model.GetDecoderBlockFfnGateWeight(index);
  param.swiglu_ffn.wdown = model.GetDecoderBlockFfnDownWeight(index);

  return param;
}
}  // namespace

Transformer::Transformer(const Model& model)
    : model_(model),
      embedding_size_(model_.GetTokenEmbeddingLength()),
      token_embeddings_(model_.GetTokenEmbeddings()),
      embeddings_(compute_engine_, embedding_size_),
      logits_(compute_engine_, model_.GetVocabSize()),
      final_rms_norm_(
          compute_engine_,
          {model_.GetOutputNormGamma(), model_.GetLayerNormEpsilon()}) {
  const int decoder_block_count = model_.GetDecoderBlockCount();
  for (int i = 0; i < decoder_block_count; i++) {
    decoder_blocks_.emplace_back(compute_engine_,
                                 GetDecoderBlockParam(model_, i));
  }
}

Transformer::~Transformer() = default;

std::span<const float> Transformer::Prefill(std::span<const Token> tokens) {
  MatrixView block_input;
  int64_t token_counts = 0;
  const int64_t chunk_size = embeddings_.Matrix().shape[0];
  for (int64_t chunk = 0; chunk < tokens.size(); chunk += chunk_size) {
    token_counts =
        std::min(chunk_size, static_cast<int64_t>(tokens.size()) - chunk);
    block_input = LookupTokenEmbeddings(tokens.subspan(chunk, token_counts));
    for (auto& decoder_block : decoder_blocks_) {
      block_input = decoder_block.Forward(block_input);
    }
  }

  // Run final output block for the last token.
  VectorView final_rms_output =
      final_rms_norm_.Forward(block_input.Bottom(1)).At(0);

  // Linear layer.
  compute_engine_.MatMul(logits_.Vector(), token_embeddings_, final_rms_output);

  compute_engine_.Softmax(logits_.Vector());

  return logits_.Vector().As<const float>();
}

std::span<const float> Transformer::Predict(Token token) {
  // Predict is just Prefill with batch size 1
  return Prefill({&token, 1});
}

MatrixView Transformer::LookupTokenEmbeddings(
    std::span<const Token> tokens) {
  for (int i = 0; i < tokens.size(); i++) {
    CHECK_NE(tokens[i], Token::INVALID);
    VectorView embedding = token_embeddings_.At(tokens[i].id);
    compute_engine_.Copy(embeddings_.Matrix().At(i), embedding);
  }

  return embeddings_.Matrix().Top(tokens.size());
}

}  // namespace tlm
