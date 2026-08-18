#include "transformer/group_query_attn_block.h"

#include "compute/compute_engine.h"

namespace tlm {

GroupQueryAttnBlock::GroupQueryAttnBlock(ComputeEngine& compute)
  : compute_(compute) {
  query_storage_ = compute_.Alloc(key_size_);
  query_ = query_storage_->AsVector(key_size_);
  k_cache_storage_ = compute_.Alloc(context_length_ * key_size_);
  k_cache_ = k_cache_storage_->AsMatrix(context_length_, key_size_);
  v_cache_storage_ = compute_.Alloc(context_length_ * value_size_);
  v_cache_ = v_cache_storage_->AsMatrix(context_length_, value_size_);
  attn_storage_ = compute_.Alloc(value_size_);
  attn_ = attn_storage_->AsVector(value_size_);
}
GroupQueryAttnBlock::~GroupQueryAttnBlock() = default;

GroupQueryAttnBlock::GroupQueryAttnBlock(GroupQueryAttnBlock&&) = default;

VectorView GroupQueryAttnBlock::Forward(VectorView input) {
  compute_.MatMul(query_, input, wq_);
  compute_.Rope(query_, token_index_);

  MutableVectorView key = k_cache_.At(token_index_);
  compute_.MatMul(key, input, wk_);
  compute_.Rope(key, token_index_);

  MutableVectorView value = v_cache_.At(token_index_);
  compute_.MatMul(value, input, wv_);

  compute_.Attn(attn_, query_.View(), k_cache_.View(), v_cache_.View());

  token_index_++;
  return attn_.View();
}

}  // namespace tlm
