#include "transformer/decoder_block.h"

#include "compute/compute_engine.h"

namespace tlm {

DecoderBlock::DecoderBlock(ComputeEngine& compute_engine)
    : compute_engine_(compute_engine),
      attn_norm_(compute_engine),
      group_query_attn_(compute_engine),
      ffn_norm_(compute_engine) {}
DecoderBlock::~DecoderBlock() = default;

DecoderBlock::DecoderBlock(DecoderBlock&&) = default;

VectorView DecoderBlock::Forward(VectorView input) {
  VectorView attn_norm_output = attn_norm_.Forward(input);
  VectorView gqa_output = group_query_attn_.Forward(attn_norm_output);

  MutableVectorView attn_output;
  compute_engine_.Add(attn_output, input, gqa_output);

  VectorView ffn_norm_output = ffn_norm_.Forward(attn_output.View());
  VectorView swiglu_ffn_output = swiglu_ffn_block_.Forward(ffn_norm_output);

  MutableVectorView decoder_block_output;
  compute_engine_.Add(decoder_block_output, attn_output.View(),
                      swiglu_ffn_output);

  return decoder_block_output.View();
}

}  // namespace tlm
