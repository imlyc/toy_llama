#include "compute/tensor_storage.h"

#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/dtype.h"
#include "tensor/tensor_view.h"

namespace tlm {
namespace {

// TensorStorage owns a buffer and a cached view into it. The pair is only
// consistent if the object is never copied (a copy would duplicate the buffer
// while the view kept pointing at the original) and if moves carry the buffer
// address along with the view.

TEST(TensorStorageTest, IsMoveOnlyType) {
  static_assert(!std::is_copy_constructible_v<MatrixStorage>);
  static_assert(!std::is_copy_assignable_v<MatrixStorage>);
  static_assert(std::is_move_constructible_v<MatrixStorage>);
  static_assert(!std::is_copy_constructible_v<VectorStorage>);
  static_assert(std::is_move_constructible_v<VectorStorage>);
}

// The view describes a contiguous F32 buffer of exactly the requested shape.
TEST(TensorStorageTest, ViewMatchesShape) {
  MatrixStorage storage({3, 4});
  const MutableMatrixView& view = storage.View();

  EXPECT_EQ(view.dtype, DType::F32);
  EXPECT_NE(view.data, nullptr);
  EXPECT_EQ(view.shape[0], 3);
  EXPECT_EQ(view.shape[1], 4);
  EXPECT_EQ(view.stride[0], 16);  // 4 floats per row
  EXPECT_EQ(view.stride[1], 4);
  EXPECT_EQ(view.TotalBytes(), 3 * 4 * 4);
  EXPECT_EQ(view.As<float>().size(), 12u);
}

TEST(TensorStorageTest, VectorStorageView) {
  VectorStorage storage({8});
  const MutableVectorView& view = storage.View();

  EXPECT_EQ(view.dtype, DType::F32);
  EXPECT_EQ(view.shape[0], 8);
  EXPECT_EQ(view.stride[0], 4);
  EXPECT_EQ(view.TotalBytes(), 32);
}

// Writes through the view land in the storage and are readable back through
// a fresh copy of the view (views are handles onto the same memory).
TEST(TensorStorageTest, ViewWritesPersist) {
  MatrixStorage storage({2, 2});
  std::span<float> w = storage.View().As<float>();
  w[0] = 1.f;
  w[3] = 4.f;

  MutableMatrixView again = storage.View();
  std::span<const float> r = again.As<const float>();
  EXPECT_FLOAT_EQ(r[0], 1.f);
  EXPECT_FLOAT_EQ(r[3], 4.f);
}

// After a move the view must point at the buffer the new object now owns,
// i.e. the same address, holding the same data. This is the invariant that a
// fallback-to-copy silently broke (the view kept pointing at a freed buffer).
TEST(TensorStorageTest, MovePreservesViewAndData) {
  MatrixStorage source({2, 3});
  source.View().As<float>()[5] = 42.f;
  const std::byte* address = source.View().data;

  MatrixStorage moved(std::move(source));

  EXPECT_EQ(moved.View().data, address);
  EXPECT_EQ(moved.View().shape[0], 2);
  EXPECT_EQ(moved.View().shape[1], 3);
  EXPECT_FLOAT_EQ(moved.View().As<const float>()[5], 42.f);
}

// The scenario that broke inference: storages held by value inside objects
// that live in a std::vector, which moves them on every reallocation. Data
// written before the growth must still be readable, through the view, after.
struct HoldsStorage {
  explicit HoldsStorage(int64_t rows) : buffer({rows, 4}) {}
  HoldsStorage(HoldsStorage&&) = default;
  MatrixStorage buffer;
};

TEST(TensorStorageTest, SurvivesVectorReallocation) {
  std::vector<HoldsStorage> holders;
  holders.emplace_back(2);
  std::span<float> first = holders[0].buffer.View().As<float>();
  for (size_t i = 0; i < first.size(); ++i) first[i] = static_cast<float>(i);
  const std::byte* address = holders[0].buffer.View().data;

  // Grow well past any initial capacity so several reallocations happen.
  for (int i = 0; i < 16; ++i) holders.emplace_back(2);

  const MutableMatrixView& view = holders[0].buffer.View();
  EXPECT_EQ(view.data, address);  // buffer moved with the object, not copied
  std::span<const float> values = view.As<const float>();
  ASSERT_EQ(values.size(), 8u);
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_FLOAT_EQ(values[i], static_cast<float>(i)) << "i=" << i;
  }
  // Every holder owns a distinct buffer.
  for (size_t i = 1; i < holders.size(); ++i) {
    EXPECT_NE(holders[i].buffer.View().data, address) << "i=" << i;
  }
}

}  // namespace
}  // namespace tlm
