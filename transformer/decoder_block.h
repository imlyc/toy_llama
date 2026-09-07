#pragma once

#include <memory>

#include "tensor/tensor_view.h"
#include "transformer/group_query_attn_block.h"
#include "transformer/rms_norm_layer.h"
#include "transformer/swiglu_ffn_block.h"

namespace tlm {
class ComputeEngine;
class Storage;

class DecoderBlock {
 public:
  struct Param {
    int embedding_size = 2048;
    RmsNormLayer::Param attn_norm;
    GroupQueryAttnBlock::Param gqa;
    RmsNormLayer::Param ffn_norm;
    SwiGluFfnBlock::Param swiglu_ffn;
  };

  DecoderBlock(ComputeEngine& compute_engine, const Param& param);
  ~DecoderBlock();

  DecoderBlock(DecoderBlock&&);

  VectorView Forward(VectorView input);

 private:
  ComputeEngine& compute_engine_;
  RmsNormLayer attn_norm_;
  GroupQueryAttnBlock group_query_attn_;
  RmsNormLayer ffn_norm_;
  SwiGluFfnBlock swiglu_ffn_block_;

  std::unique_ptr<Storage> attn_output_storage_;
  MutableVectorView attn_output_;

  std::unique_ptr<Storage> decoder_block_output_storage_;
  MutableVectorView decoder_block_output_;
};

}  // namespace tlm
