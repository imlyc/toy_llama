#pragma once

#include <algorithm>
#include <array>
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
  std::array<int64_t, Rank> shape {};

  // Size in bytes of stride in each dimension, ordered in C style.
  // stride[Rank-1] is the bytes per block.
  std::array<int64_t, Rank> stride {};

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

  TensorView<Rank - 1, Mutable> At(int64_t index) const requires (Rank >= 2) {
    TensorView<Rank - 1, Mutable> ret;
    ret.dtype = dtype;
    ret.data = data + stride[0] * index;
    std::copy_n(shape.begin() + 1, Rank - 1, ret.shape.begin());
    std::copy_n(stride.begin() + 1, Rank - 1, ret.stride.begin());
    return ret;
  }

  // Return an immutable view of the current tensor.
  TensorView<Rank, false> View() const {
    return {dtype, data, shape, stride};
  }
};

using VectorView = TensorView<1>;
using MatrixView = TensorView<2>;
using Tensor3View = TensorView<3>;

using MutableVectorView = TensorView<1, true>;
using MutableMatrixView = TensorView<2, true>;
using MutableTensor3View = TensorView<3, true>;

}  // namespace tlm
