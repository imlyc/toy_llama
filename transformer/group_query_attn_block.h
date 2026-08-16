#pragma once

#include "tensor/tensor_view.h"
#include "transformer/rope_layer.h"

namespace tlm {
class ComputeEngine;

class GroupQueryAttnBlock {
 public:
  explicit GroupQueryAttnBlock(ComputeEngine& compute);
  ~GroupQueryAttnBlock();

  VectorView Forward(VectorView input);

 private:
  ComputeEngine& compute_;

  MatrixView wq_;
  MatrixView wk_;
  MatrixView wv_;

  RopeLayer q_rope_layer_;
  RopeLayer k_rope_layer_;
};

}  // namespace tlm
