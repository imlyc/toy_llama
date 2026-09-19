#include "transformer/tensor_storage.h"

#include "compute/compute_engine.h"
#include "compute/storage.h"

namespace tlm {
namespace {
constexpr int kMatrixChunkRowCount = 512;
}  // namespace

MatrixStorage::MatrixStorage(ComputeEngine& compute, int64_t row, int64_t col)
    : storage_(compute.Alloc(row * col)),
      view_(storage_->AsMatrix(row, col)) {}

MatrixStorage::MatrixStorage(ComputeEngine& compute, int64_t col)
    : MatrixStorage(compute, kMatrixChunkRowCount, col) {}

MatrixStorage::~MatrixStorage() = default;

MatrixStorage::MatrixStorage(MatrixStorage&&) = default;

MutableMatrixView& MatrixStorage::Matrix() {
  return view_;
}

VectorStorage::VectorStorage(ComputeEngine& compute, int64_t size)
  : storage_(compute.Alloc(size)),
    view_(storage_->AsVector(size)) {}

VectorStorage::~VectorStorage() = default;

VectorStorage::VectorStorage(VectorStorage&&) = default;

MutableVectorView& VectorStorage::Vector() {
  return view_;
}

}  // namespace tlm
