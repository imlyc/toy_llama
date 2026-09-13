#pragma once

#include "tensor/tensor_view.h"
#include "transformer/tensor_storage.h"

namespace tlm {
class ComputeEngine;
class Storage;

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
