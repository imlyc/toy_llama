#pragma once

#include "tensor/tensor_view.h"

namespace tlm {

class RopeLayer {
 public:
  RopeLayer();
  ~RopeLayer();

  VectorView Forward(VectorView input);
};

}  // namespace tlm
