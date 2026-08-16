#pragma once

#include "tensor/tensor_view.h"

namespace tlm {

class ComputeEngine {
 public:
  ComputeEngine();
  ~ComputeEngine();

  ComputeEngine(const ComputeEngine&) = delete;
  ComputeEngine& operator=(const ComputeEngine&) = delete;

  // The return value should be a storage type, or the caller should pass in the
  // output MutableVectorView.
  VectorView Add(VectorView lhs, VectorView rhs);
  VectorView MatMul(VectorView lhs, MatrixView rhs);
};

}  // namespace tlm
