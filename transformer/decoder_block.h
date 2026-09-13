#pragma once

#include <memory>

#include "tensor/tensor_view.h"
#include "transformer/group_query_attn_block.h"
#include "transformer/rms_norm_layer.h"
#include "transformer/swiglu_ffn_block.h"
#include "transformer/tensor_storage.h"

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

  MatrixView Forward(MatrixView input);

 private:
  ComputeEngine& compute_engine_;
  RmsNormLayer attn_norm_;
  GroupQueryAttnBlock group_query_attn_;
  RmsNormLayer ffn_norm_;
  SwiGluFfnBlock swiglu_ffn_block_;

  MatrixStorage attn_output_;
  MatrixStorage decoder_block_output_;
};

}  // namespace tlm
