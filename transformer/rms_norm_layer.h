#pragma once

#include "tensor/tensor_view.h"
#include "transformer/tensor_storage.h"

namespace tlm {
class ComputeEngine;
class Storage;

class RmsNormLayer {
 public:
  struct Param {
    VectorView gamma;
    float epsilon = 0;
  };

  RmsNormLayer(ComputeEngine& compute, const Param& param);
  ~RmsNormLayer();

  RmsNormLayer(RmsNormLayer&&);

  MatrixView Forward(MatrixView input);

 private:
  ComputeEngine& compute_;
  const VectorView gamma_;
  const float epsilon_;

  MatrixStorage output_;
};

}  // namespace tlm
