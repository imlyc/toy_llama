#include "transformer/linear_layer.h"

namespace tlm {

LinearLayer::LinearLayer() = default;
LinearLayer::~LinearLayer() = default;

VectorView LinearLayer::Forward(VectorView input) {
  return {};
}

}  // namespace tlm
