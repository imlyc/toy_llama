#include "compute/compute_engine.h"

#include "compute/storage.h"

namespace tlm {

ComputeEngine::ComputeEngine() = default;
ComputeEngine::~ComputeEngine() = default;

std::unique_ptr<Storage> ComputeEngine::Alloc(int64_t size) {
  return std::make_unique<Storage>(size);
}

void ComputeEngine::Add(MutableVectorView out, VectorView lhs, VectorView rhs) {
}

void ComputeEngine::MatMul(MutableVectorView out,
                           VectorView lhs,
                           MatrixView rhs) {}

void ComputeEngine::Attn(MutableVectorView out,
                         VectorView q,
                         MatrixView k,
                         MatrixView v) {}

void ComputeEngine::Rope(MutableVectorView view, int64_t position) {}

}  // namespace tlm
