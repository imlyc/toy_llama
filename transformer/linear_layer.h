#pragma once

#include "tensor/tensor_view.h"

namespace tlm {

class LinearLayer {
 public:
  LinearLayer();
  ~LinearLayer();

  VectorView Forward(VectorView input);
};

}  // namespace tlm
