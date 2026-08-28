#pragma once

#include <memory>

#include "compute/storage.h"
#include "tensor/tensor_view.h"

namespace tlm {
class ComputeEngine;

class GroupQueryAttnBlock {
 public:
  explicit GroupQueryAttnBlock(ComputeEngine& compute);
  ~GroupQueryAttnBlock();

  GroupQueryAttnBlock(GroupQueryAttnBlock&&);

  VectorView Forward(VectorView input);

 private:
  ComputeEngine& compute_;

  // TODO: Use parameters from model.
  const int64_t key_size_ = 64;
  const int64_t value_size_ = 64;
  const int64_t context_length_ = 131072;
  const int64_t head_count_ = 32;
  const int64_t head_count_kv_ = 8;
  const int64_t output_size_ = 2048;
  const float rope_freq_base_ = 500000;
  const int rope_dimension_count_ = 64;

  int64_t token_index_ = 0;

  MatrixView wq_;
  MatrixView wk_;
  MatrixView wv_;
  MatrixView wo_;

  std::unique_ptr<Storage> query_storage_;
  MutableVectorView query_;

  std::unique_ptr<Storage> k_cache_storage_;
  MutableMatrixView k_cache_;

  std::unique_ptr<Storage> v_cache_storage_;
  MutableMatrixView v_cache_;

  std::unique_ptr<Storage> attn_storage_;
  MutableVectorView attn_;

  std::unique_ptr<Storage> output_storage_;
  MutableVectorView output_;
};

}  // namespace tlm
