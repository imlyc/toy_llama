#include "transformer/tensor_storage.h"

#include "compute/compute_engine.h"
#include "compute/storage.h"

namespace tlm {
namespace {
constexpr int kInitialMatrixRowCount = 32;
}  // namespace

MatrixStorage::MatrixStorage(ComputeEngine& compute, int64_t row, int64_t col)
    : compute_(compute),
      storage_(compute_.Alloc(row * col)),
      view_(storage_->AsMatrix(row, col)) {}

MatrixStorage::MatrixStorage(ComputeEngine& compute, int64_t col)
    : MatrixStorage(compute, kInitialMatrixRowCount, col) {}

MatrixStorage::~MatrixStorage() = default;

MatrixStorage::MatrixStorage(MatrixStorage&&) = default;

void MatrixStorage::Reset(int64_t row) {
  if (row > view_.shape[0]) {
    storage_ = compute_.Alloc(row * view_.shape[1]);
  }

  view_ = storage_->AsMatrix(row, view_.shape[1]);
}

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
