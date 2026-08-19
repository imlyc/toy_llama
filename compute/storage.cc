#include "compute/storage.h"

#include <glog/logging.h>

#include "tensor/dtype.h"

namespace tlm {

Storage::Storage(int64_t size) : data_(size * sizeof(float)) {}
Storage::~Storage() = default;

MutableVectorView Storage::AsVector(int64_t size) {
  MutableVectorView view =
      MutableVectorView::Create(DType::F32, data_.data(), size);
  CHECK_LE(view.TotalBytes(), data_.size());
  return view;
}

MutableMatrixView Storage::AsMatrix(int64_t row, int64_t col) {
  MutableMatrixView view =
      MutableMatrixView::Create(DType::F32, data_.data(), row, col);
  CHECK_LE(view.TotalBytes(), data_.size());
  return view;
}
}  // namespace tlm
