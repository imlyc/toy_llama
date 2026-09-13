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

  // Copy data from `src` to `dst`. `dst` is a memory allocated by Alloc.
  void Copy(MutableVectorView dst, VectorView src);

  void Add(MutableMatrixView out, MatrixView lhs, MatrixView rhs);
  void MatMul(MutableVectorView out, MatrixView lhs, VectorView rhs);

  // out = lhs @ transpose(rhs)
  void MatMulT(MutableMatrixView out, MatrixView lhs, MatrixView rhs);

  // Scaled dot product attention. A = softmax(Q @ transpose(K) / sqrt(dk)) @ V.
  // Support different head count for GQA, MHA.
  void Attn(MutableMatrixView out, MatrixView q, MatrixView k, MatrixView v,
            int64_t head_count_q, int64_t head_count_kv);

  void RmsNorm(MutableMatrixView out,
               MatrixView input,
               VectorView gamma,
               float epsilon);

  // In place swiglu gate + elementwise mul up
  // gate = swiglu(gate) mul up
  void SwiGluMul(MutableMatrixView gate, MatrixView up);

  // In place softmax.
  void Softmax(MutableVectorView view);

  // In place RoPE.
  void Rope(MutableMatrixView view,
            int64_t position,
            float freq_base,
            int dimension_count,
            VectorView rope_freqs);
};

}  // namespace tlm
