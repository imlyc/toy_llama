#include "compute/storage.h"

#include <glog/logging.h>

#include "tensor/dtype.h"

namespace tlm {

Storage::Storage(int64_t size) : data_(size * sizeof(float)) {}
Storage::~Storage() = default;

MutableVectorView Storage::AsVector(int64_t size) {
  MutableVectorView view;
  view.dtype = DType::F32;
  view.data = data_.data();
  view.shape[0] = size;
  view.stride[0] = sizeof(float);
  CHECK_LE(view.TotalBytes(), data_.size());
  return view;
}

MutableMatrixView Storage::AsMatrix(int64_t row, int64_t col) {
  MutableMatrixView view;
  view.dtype = DType::F32;
  view.data = data_.data();
  view.shape = {row, col};
  view.stride = {static_cast<int64_t>(col * sizeof(float)), sizeof(float)};
  CHECK_LE(view.TotalBytes(), data_.size());
  return view;
}
}  // namespace tlm
