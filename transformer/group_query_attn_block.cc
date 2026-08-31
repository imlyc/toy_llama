#include "transformer/group_query_attn_block.h"

#include "compute/compute_engine.h"

namespace tlm {

GroupQueryAttnBlock::GroupQueryAttnBlock(ComputeEngine& compute)
    : compute_(compute) {
  query_storage_ = compute_.Alloc(key_size_ * head_count_);
  query_ = query_storage_->AsVector(key_size_ * head_count_);
  k_cache_storage_ =
      compute_.Alloc(context_length_ * key_size_ * head_count_kv_);
  k_cache_ =
      k_cache_storage_->AsMatrix(context_length_, key_size_ * head_count_kv_);
  v_cache_storage_ =
      compute_.Alloc(context_length_ * value_size_ * head_count_kv_);
  v_cache_ =
      v_cache_storage_->AsMatrix(context_length_, value_size_ * head_count_kv_);
  attn_storage_ = compute_.Alloc(value_size_ * head_count_);
  attn_ = attn_storage_->AsVector(value_size_ * head_count_);
  output_storage_ = compute_.Alloc(output_size_);
  output_ = output_storage_->AsVector(output_size_);
}
GroupQueryAttnBlock::~GroupQueryAttnBlock() = default;

GroupQueryAttnBlock::GroupQueryAttnBlock(GroupQueryAttnBlock&&) = default;

VectorView GroupQueryAttnBlock::Forward(VectorView input) {
  compute_.MatMul(query_, wq_, input);
  compute_.Rope(query_, token_index_, rope_freq_base_, rope_dimension_count_);

  MutableVectorView key = k_cache_.At(token_index_);
  compute_.MatMul(key, wk_, input);
  compute_.Rope(key, token_index_, rope_freq_base_, rope_dimension_count_);

  MutableVectorView value = v_cache_.At(token_index_);
  compute_.MatMul(value, wv_, input);

  int64_t length = token_index_ + 1;
  compute_.Attn(attn_, query_.View(), k_cache_.Top(length),
                v_cache_.Top(length), head_count_, head_count_kv_);

  compute_.MatMul(output_, wo_, attn_.View());

  token_index_++;
  return output_.View();
}

}  // namespace tlm
