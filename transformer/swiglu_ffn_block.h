#pragma once

#include "tensor/tensor_view.h"

namespace tlm {
class ComputeEngine;
class Storage;

class SwiGluFfnBlock {
 public:
  explicit SwiGluFfnBlock(ComputeEngine& compute);
  ~SwiGluFfnBlock();

  SwiGluFfnBlock(SwiGluFfnBlock&&);

  VectorView Forward(VectorView input);

 private:
  ComputeEngine& compute_;

  MatrixView wup_;
  MatrixView wgate_;
  MatrixView wdown_;

  std::unique_ptr<Storage> up_storage_;
  MutableVectorView up_;

  std::unique_ptr<Storage> gate_storage_;
  MutableVectorView gate_;

  std::unique_ptr<Storage> output_storage_;
  MutableVectorView output_;
};

}  // namespace tlm
