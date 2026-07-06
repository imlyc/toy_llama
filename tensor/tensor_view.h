#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "tensor/dtype.h"

namespace tlm {

// View class of a tensor. The actual data is owned by a different class, please
// make sure the storage lives longer than the view.
//
// Terminologies:
// 1. element: one mathematical value.
// 2. block: smallest self-contained encoded chunk — for F32 a 4-byte float
// encoding 1 element; for Q8_0 a 34-byte struct (f16 scale + 32 int8) encoding
// 32 elements.
template <int Rank, bool Mutable = false>
struct TensorView {
  using Byte = std::conditional_t<Mutable, std::byte, const std::byte>;

  DType dtype = DType::F32;
  Byte* data = nullptr;

  // Number of elements in each dimension, ordered in C style.
  int64_t shape[Rank] = {0};

  // Size in bytes of stride in each dimension, ordered in C style.
  // stride[Rank-1] is the bytes per block.
  int64_t stride[Rank] = {0};

  int64_t TotalBytes() const {
    if constexpr (Rank == 1) {
        return shape[0] / GetDTypeBlockElements(dtype) * stride[0];
    } else {
        return shape[0] * stride[0];
    }
  }

  // View the data as specific type.
  template <typename T>
  std::span<T> As() const {
    return std::span(reinterpret_cast<T*>(data), TotalBytes() / sizeof(T));
  }
};

using VectorView = TensorView<1>;
using MatrixView = TensorView<2>;
using Tensor3View = TensorView<3>;

using MutableVectorView = TensorView<1, true>;
using MutableMatrixView = TensorView<2, true>;
using MutableTensor3View = TensorView<3, true>;

}  // namespace tlm
