#pragma once

#include "tensor/tensor_view.h"

namespace tlm {
class ComputeEngine;
class Storage;

class RmsNormLayer {
 public:
  explicit RmsNormLayer(ComputeEngine& compute);
  ~RmsNormLayer();

  RmsNormLayer(RmsNormLayer&&);

  VectorView Forward(VectorView input);

 private:
  ComputeEngine& compute_;
  VectorView gamma_;
  const float epsilon_ = 0.000009999999747378752;

  std::unique_ptr<Storage> output_storage_;
  MutableVectorView output_;
};

}  // namespace tlm
