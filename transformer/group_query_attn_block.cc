#include "transformer/group_query_attn_block.h"

#include "compute/compute_engine.h"

namespace tlm {

GroupQueryAttnBlock::GroupQueryAttnBlock(ComputeEngine& compute,
                                         const Param& param)
    : compute_(compute),
      head_count_(param.head_count),
      head_count_kv_(param.head_count_kv),
      rope_freq_base_(param.rope_freq_base),
      rope_dimension_count_(param.rope_dimension_count),
      wq_(param.wq),
      wk_(param.wk),
      wv_(param.wv),
      wo_(param.wo),
      rope_freqs_(param.rope_freqs) {
  query_storage_ = compute_.Alloc(param.key_size * head_count_);
  query_ = query_storage_->AsVector(param.key_size * head_count_);
  k_cache_storage_ =
      compute_.Alloc(param.context_length * param.key_size * head_count_kv_);
  k_cache_ = k_cache_storage_->AsMatrix(param.context_length,
                                        param.key_size * head_count_kv_);
  v_cache_storage_ =
      compute_.Alloc(param.context_length * param.value_size * head_count_kv_);
  v_cache_ = v_cache_storage_->AsMatrix(param.context_length,
                                        param.value_size * head_count_kv_);
  attn_storage_ = compute_.Alloc(param.value_size * head_count_);
  attn_ = attn_storage_->AsVector(param.value_size * head_count_);
  output_storage_ = compute_.Alloc(wo_.shape[0]);
  output_ = output_storage_->AsVector(wo_.shape[0]);
}
GroupQueryAttnBlock::~GroupQueryAttnBlock() = default;

GroupQueryAttnBlock::GroupQueryAttnBlock(GroupQueryAttnBlock&&) = default;

VectorView GroupQueryAttnBlock::Forward(VectorView input) {
  compute_.MatMul(query_, wq_, input);
  compute_.Rope(query_, token_index_, rope_freq_base_, rope_dimension_count_,
                rope_freqs_);

  MutableVectorView key = k_cache_.At(token_index_);
  compute_.MatMul(key, wk_, input);
  compute_.Rope(key, token_index_, rope_freq_base_, rope_dimension_count_,
                rope_freqs_);

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
