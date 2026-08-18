#pragma once

#include "tensor/tensor_view.h"

namespace tlm {

// Storage or memory for the actual data during computation.
class Storage {
 public:
  explicit Storage(int64_t size);
  ~Storage();

  MutableVectorView AsVector(int64_t size);
  MutableMatrixView AsMatrix(int64_t row, int64_t col);

 private:
  std::vector<std::byte> data_;
};

}  // namespace tlm
