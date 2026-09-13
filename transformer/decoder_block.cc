#include "transformer/decoder_block.h"

#include "compute/compute_engine.h"

namespace tlm {

DecoderBlock::DecoderBlock(ComputeEngine& compute_engine, const Param& param)
    : compute_engine_(compute_engine),
      attn_norm_(compute_engine, param.attn_norm),
      group_query_attn_(compute_engine, param.gqa),
      ffn_norm_(compute_engine, param.ffn_norm),
      swiglu_ffn_block_(compute_engine, param.swiglu_ffn),
      attn_output_(compute_engine_, param.embedding_size),
      decoder_block_output_(compute_engine_, param.embedding_size) {}
DecoderBlock::~DecoderBlock() = default;

DecoderBlock::DecoderBlock(DecoderBlock&&) = default;

MatrixView DecoderBlock::Forward(MatrixView input) {
  attn_output_.Reset(input.shape[0]);
  decoder_block_output_.Reset(input.shape[0]);

  MatrixView attn_norm_output = attn_norm_.Forward(input);
  MatrixView gqa_output = group_query_attn_.Forward(attn_norm_output);

  compute_engine_.Add(attn_output_.Matrix(), input, gqa_output);

  MatrixView ffn_norm_output = ffn_norm_.Forward(attn_output_.Matrix().View());
  MatrixView swiglu_ffn_output = swiglu_ffn_block_.Forward(ffn_norm_output);

  compute_engine_.Add(decoder_block_output_.Matrix(),
                      attn_output_.Matrix().View(), swiglu_ffn_output);

  return decoder_block_output_.Matrix().View();
}

}  // namespace tlm
