#pragma once

#include "tensor/tensor_view.h"

namespace tlm {

class SwiGluFfnBlock {
 public:
  SwiGluFfnBlock();
  ~SwiGluFfnBlock();

  VectorView Forward(VectorView input);
};

}  // namespace tlm
