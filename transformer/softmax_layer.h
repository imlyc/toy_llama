#pragma once

#include "tensor/tensor_view.h"

namespace tlm {

class SoftmaxLayer {
 public:
  SoftmaxLayer();
  ~SoftmaxLayer();

  VectorView Forward(VectorView input);
};

}  // namespace tlm
