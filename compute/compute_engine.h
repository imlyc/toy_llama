#pragma once

#include <memory>

#include "tensor/tensor_view.h"

namespace tlm {
class Storage;

class ComputeEngine {
 public:
  ComputeEngine();
  ~ComputeEngine();

  ComputeEngine(const ComputeEngine&) = delete;
  ComputeEngine& operator=(const ComputeEngine&) = delete;

  std::unique_ptr<Storage> Alloc(int64_t size);

  void Add(MutableVectorView out, VectorView lhs, VectorView rhs);
  void MatMul(MutableVectorView out, VectorView lhs, MatrixView rhs);

  // Scaled dot product attention. a = softmax(q @ transpose(K) / sqrt(dk)) @ V.
  void Attn(MutableVectorView out, VectorView q, MatrixView k, MatrixView v);

  // In place RoPE.
  void Rope(MutableVectorView view, int64_t position);
};

}  // namespace tlm
