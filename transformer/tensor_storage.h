#pragma once

#include <memory>

#include "tensor/tensor_view.h"

namespace tlm {
class ComputeEngine;
class Storage;

// Storage for matrix with updatable row but fixed col.
class MatrixStorage {
 public:
  MatrixStorage(ComputeEngine& compute, int64_t row, int64_t col);
  MatrixStorage(ComputeEngine& compute, int64_t col);
  ~MatrixStorage();

  MatrixStorage(MatrixStorage&&);

  MutableMatrixView& Matrix();

 private:
  ComputeEngine& compute_;
  std::unique_ptr<Storage> storage_;
  MutableMatrixView view_;
};

class VectorStorage {
 public:
  VectorStorage(ComputeEngine& compute, int64_t size);
  ~VectorStorage();

  VectorStorage(VectorStorage&&);

  MutableVectorView& Vector();
  
 private:
  std::unique_ptr<Storage> storage_;
  MutableVectorView view_;
};

}  // namespace tlm
