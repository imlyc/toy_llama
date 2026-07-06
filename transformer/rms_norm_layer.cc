#include "transformer/rms_norm_layer.h"

namespace tlm {

RmsNormLayer::RmsNormLayer() = default;
RmsNormLayer::~RmsNormLayer() = default;

VectorView RmsNormLayer::Forward(VectorView input) {
  return {};
}

}  // namespace tlm
