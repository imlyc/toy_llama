#pragma once

#include <array>
#include <functional>
#include <numeric>
#include <vector>

#include "tensor/tensor_view.h"

namespace tlm {

// Storage or memory for the actual data during computation.
template <int Rank>
class TensorStorage {
 public:
  explicit TensorStorage(const std::array<int64_t, Rank>& shape)
      : data_(std::accumulate(shape.begin(),
                              shape.end(),
                              static_cast<int64_t>(1),
                              std::multiplies<int64_t>()) *
              sizeof(float)),
        view_(TensorView<Rank, true>::Create(DType::F32, data_.data(), shape)) {
  }

  // This class manages a large chunk of memory, we should avoid copy it.
  TensorStorage(const TensorStorage&) = delete;

  // The move constructor is necessary otherwise view_ points to an old memory
  // after copy (no default move constructor will fallback to copy constructor).
  TensorStorage(TensorStorage&&) = default;

  ~TensorStorage() = default;

  const TensorView<Rank, true>& View() const {
    return view_;
  }

 private:
  std::vector<std::byte> data_;
  TensorView<Rank, true> view_;
};

using VectorStorage = TensorStorage<1>;
using MatrixStorage = TensorStorage<2>;

}  // namespace tlm
