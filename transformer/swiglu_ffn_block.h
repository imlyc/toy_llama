#pragma once

#include "compute/tensor_storage.h"
#include "tensor/tensor_view.h"

namespace tlm {
class ComputeEngine;

class SwiGluFfnBlock {
 public:
  struct Param {
    MatrixView wup;
    MatrixView wgate;
    MatrixView wdown;
  };

  SwiGluFfnBlock(ComputeEngine& compute, const Param& param);
  ~SwiGluFfnBlock();

  SwiGluFfnBlock(SwiGluFfnBlock&&);

  MatrixView Forward(MatrixView input);

 private:
  ComputeEngine& compute_;

  MatrixView wup_;
  MatrixView wgate_;
  MatrixView wdown_;

  MatrixStorage up_;
  MatrixStorage gate_;
  MatrixStorage output_;
};

}  // namespace tlm
