#pragma once

#include "tensor/tensor_view.h"

namespace tlm {

class RmsNormLayer {
 public:
  RmsNormLayer();
  ~RmsNormLayer();

  VectorView Forward(VectorView input);
};

}  // namespace tlm
