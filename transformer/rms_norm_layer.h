#pragma once

#include "tensor/tensor_view.h"

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

  VectorView Forward(VectorView input);

 private:
  ComputeEngine& compute_;
  const VectorView gamma_;
  const float epsilon_;

  std::unique_ptr<Storage> output_storage_;
  MutableVectorView output_;
};

}  // namespace tlm
