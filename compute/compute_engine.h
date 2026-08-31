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
  void MatMul(MutableVectorView out, MatrixView lhs, VectorView rhs);

  // Scaled dot product attention. a = softmax(q @ transpose(K) / sqrt(dk)) @ V.
  // Support different head count for GQA, MHA.
  void Attn(MutableVectorView out, VectorView q, MatrixView k, MatrixView v,
            int64_t head_count_q, int64_t head_count_kv);

  // In place softmax.
  void Softmax(MutableVectorView view);

  // In place RoPE.
  void Rope(MutableVectorView view,
            int64_t position,
            float freq_base,
            int dimension_count);
};

}  // namespace tlm
