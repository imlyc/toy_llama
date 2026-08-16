#include "compute/compute_engine.h"

namespace tlm {

ComputeEngine::ComputeEngine() = default;
ComputeEngine::~ComputeEngine() = default;

VectorView ComputeEngine::Add(VectorView lhs, VectorView rhs) {
  return {};
}

VectorView ComputeEngine::MatMul(VectorView lhs, MatrixView rhs) {
  return {};
}

}  // namespace tlm
