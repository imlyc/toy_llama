#pragma once

#include "tensor/tensor_view.h"

namespace tlm {

class DecoderBlock {
 public:
  DecoderBlock();
  ~DecoderBlock();

  VectorView Forward(VectorView input);
};

}  // namespace tlm
