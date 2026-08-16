#pragma once

#include "tensor/tensor_view.h"
#include "transformer/group_query_attn_block.h"
#include "transformer/rms_norm_layer.h"
#include "transformer/swiglu_ffn_block.h"

namespace tlm {
class ComputeEngine;

class DecoderBlock {
 public:
  explicit DecoderBlock(ComputeEngine& compute_engine);
  ~DecoderBlock();

  VectorView Forward(VectorView input);

 private:
  ComputeEngine& compute_engine_;
  RmsNormLayer attn_norm_;
  GroupQueryAttnBlock group_query_attn_;
  RmsNormLayer ffn_norm_;
  SwiGluFfnBlock swiglu_ffn_block_;
};

}  // namespace tlm
