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
      rope_freqs_(param.rope_freqs),
      query_(compute_.AllocMatrix(param.key_size * head_count_)),
      k_cache_(compute_.AllocMatrix(param.context_length,
                                    param.key_size * head_count_kv_)),
      v_cache_(compute_.AllocMatrix(param.context_length,
                                    param.value_size * head_count_kv_)),
      attn_(compute_.AllocMatrix(param.value_size * head_count_)),
      output_(compute_.AllocMatrix(wo_.shape[0])) {}
GroupQueryAttnBlock::~GroupQueryAttnBlock() = default;

GroupQueryAttnBlock::GroupQueryAttnBlock(GroupQueryAttnBlock&&) = default;

MatrixView GroupQueryAttnBlock::Forward(MatrixView input) {
  MutableMatrixView query = query_.View().Top(input.shape[0]);
  MutableMatrixView attn = attn_.View().Top(input.shape[0]);
  MutableMatrixView output = output_.View().Top(input.shape[0]);

  compute_.MatMulT(query, input, wq_);
  compute_.Rope(query, token_index_, rope_freq_base_, rope_dimension_count_,
                rope_freqs_);

  MutableMatrixView key = k_cache_.View().Slice(token_index_, input.shape[0]);
  compute_.MatMulT(key, input, wk_);
  compute_.Rope(key, token_index_, rope_freq_base_, rope_dimension_count_,
                rope_freqs_);

  MutableMatrixView value =
      v_cache_.View().Slice(token_index_, input.shape[0]);
  compute_.MatMulT(value, input, wv_);

  int64_t length = token_index_ + input.shape[0];
  compute_.Attn(attn, query, k_cache_.View().Top(length),
                v_cache_.View().Top(length), head_count_, head_count_kv_);

  compute_.MatMulT(output, attn, wo_);

  token_index_ += input.shape[0];
  return output;
}

}  // namespace tlm
