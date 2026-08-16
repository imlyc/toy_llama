#include "transformer/group_query_attn_block.h"

#include "compute/compute_engine.h"

namespace tlm {

GroupQueryAttnBlock::GroupQueryAttnBlock(ComputeEngine& compute)
  : compute_(compute) {}
GroupQueryAttnBlock::~GroupQueryAttnBlock() = default;

VectorView GroupQueryAttnBlock::Forward(VectorView input) {
  VectorView query = q_rope_layer_.Forward(compute_.MatMul(input, wq_));
  VectorView key = k_rope_layer_.Forward(compute_.MatMul(input, wk_));
  VectorView value = compute_.MatMul(input, wv_);
  return {};
}

}  // namespace tlm
