#pragma once

#include "compute/tensor_storage.h"
#include "tensor/tensor_view.h"

namespace tlm {
class ComputeEngine;

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
