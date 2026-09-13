#pragma once

#include <memory>

#include "compute/storage.h"
#include "tensor/tensor_view.h"
#include "transformer/tensor_storage.h"

namespace tlm {
class ComputeEngine;

class GroupQueryAttnBlock {
 public:
  struct Param {
    int key_size = 64;
    int value_size = 64;
    int context_length = 131072;
    int head_count = 32;
    int head_count_kv = 8;
    float rope_freq_base = 500000;
    int rope_dimension_count = 64;
    MatrixView wq;
    MatrixView wk;
    MatrixView wv;
    MatrixView wo;
    VectorView rope_freqs;
  };

  GroupQueryAttnBlock(ComputeEngine& compute, const Param& param);
  ~GroupQueryAttnBlock();

  GroupQueryAttnBlock(GroupQueryAttnBlock&&);

  MatrixView Forward(MatrixView input);

 private:
  ComputeEngine& compute_;

  const int head_count_;
  const int head_count_kv_;
  const float rope_freq_base_;
  const int rope_dimension_count_;

  int64_t token_index_ = 0;

  MatrixView wq_;
  MatrixView wk_;
  MatrixView wv_;
  MatrixView wo_;
  VectorView rope_freqs_;

  MatrixStorage query_;
  MatrixStorage k_cache_;
  MatrixStorage v_cache_;
  MatrixStorage attn_;
  MatrixStorage output_;
};

}  // namespace tlm
