#include "transformer/rms_norm_layer.h"

#include "compute/compute_engine.h"

namespace tlm {

RmsNormLayer::RmsNormLayer(ComputeEngine& compute,
                           const Param& param)
    : compute_(compute),
      gamma_(param.gamma),
      epsilon_(param.epsilon),
      output_(compute_.AllocMatrix(gamma_.shape[0])) {}
RmsNormLayer::~RmsNormLayer() = default;

RmsNormLayer::RmsNormLayer(RmsNormLayer&&) = default;

MatrixView RmsNormLayer::Forward(MatrixView input) {
  MutableMatrixView output = output_.View().Top(input.shape[0]);
  compute_.RmsNorm(output, input, gamma_, epsilon_);
  return output;
}

}  // namespace tlm
