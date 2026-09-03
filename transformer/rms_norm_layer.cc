#include "transformer/rms_norm_layer.h"

#include "compute/compute_engine.h"
#include "compute/storage.h"

namespace tlm {

RmsNormLayer::RmsNormLayer(ComputeEngine& compute) : compute_(compute) {
  output_storage_ = compute_.Alloc(gamma_.shape[0]);
  output_ = output_storage_->AsVector(gamma_.shape[0]);
}
RmsNormLayer::~RmsNormLayer() = default;

RmsNormLayer::RmsNormLayer(RmsNormLayer&&) = default;

VectorView RmsNormLayer::Forward(VectorView input) {
  compute_.RmsNorm(output_, input, gamma_, epsilon_);
  return output_.View();
}

}  // namespace tlm
