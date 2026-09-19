#include "tensor/tensor_view.h"

#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/dtype.h"

namespace tlm {
namespace {

// A contiguous, row-major [2, 3, 4] F32 tensor whose element (i, j, k) holds the
// value of its flat index i*12 + j*4 + k. F32 => 1 element per block, 4 bytes
// per block, so the C-order byte strides are {48, 16, 4}.
class Tensor3F32Fixture : public ::testing::Test {
 protected:
  void SetUp() override {
    data_.resize(2 * 3 * 4);
    for (int n = 0; n < static_cast<int>(data_.size()); ++n) {
      data_[n] = static_cast<float>(n);
    }
  }

  const std::byte* base() const {
    return reinterpret_cast<const std::byte*>(data_.data());
  }
  std::byte* mutable_base() {
    return reinterpret_cast<std::byte*>(data_.data());
  }

  Tensor3View View() const {
    return Tensor3View::Create(DType::F32, base(), 2, 3, 4);
  }
  MutableTensor3View MutableView() {
    return MutableTensor3View::Create(DType::F32, mutable_base(), 2, 3, 4);
  }

  std::vector<float> data_;
};

// At on a rank-3 view yields the n-th rank-2 sub-matrix, dropping the outer dim.
TEST_F(Tensor3F32Fixture, At3DReturnsMatrixMetadata) {
  auto m = View().At(1);

  EXPECT_EQ(m.dtype, DType::F32);
  // Offset = stride[0] * index = 48 * 1 bytes = element 12.
  EXPECT_EQ(m.data, base() + 48);
  // Outer dim dropped; inner dims/strides shift down by one.
  EXPECT_EQ(m.shape[0], 3);
  EXPECT_EQ(m.shape[1], 4);
  EXPECT_EQ(m.stride[0], 16);
  EXPECT_EQ(m.stride[1], 4);
}

// At on a rank-2 view yields the n-th rank-1 row.
TEST_F(Tensor3F32Fixture, At2DReturnsVectorMetadata) {
  auto v = View().At(1).At(2);  // row (1, 2, :)

  EXPECT_EQ(v.dtype, DType::F32);
  // base + 48 (i=1) + 16*2 (j=2) = base + 80 bytes = element 20.
  EXPECT_EQ(v.data, base() + 80);
  EXPECT_EQ(v.shape[0], 4);
  EXPECT_EQ(v.stride[0], 4);
}

// Index 0 must not move the data pointer.
TEST_F(Tensor3F32Fixture, AtZeroKeepsBasePointer) {
  auto m = View().At(0);
  EXPECT_EQ(m.data, base());
}

// The last valid outer index lands on the final sub-matrix.
TEST_F(Tensor3F32Fixture, AtLastIndex) {
  auto m = View().At(1);  // i = shape[0]-1
  EXPECT_EQ(m.data, base() + 48);
}

// Chained At + As<const float>() decodes the exact underlying row.
TEST_F(Tensor3F32Fixture, AtChainedReadsCorrectRow) {
  auto row = View().At(1).At(2).As<const float>();  // expect {20, 21, 22, 23}

  ASSERT_EQ(row.size(), 4u);
  EXPECT_FLOAT_EQ(row[0], 20.f);
  EXPECT_FLOAT_EQ(row[1], 21.f);
  EXPECT_FLOAT_EQ(row[2], 22.f);
  EXPECT_FLOAT_EQ(row[3], 23.f);
}

// The first row of the first matrix is elements {0, 1, 2, 3}.
TEST_F(Tensor3F32Fixture, AtFirstRow) {
  auto row = View().At(0).At(0).As<const float>();

  ASSERT_EQ(row.size(), 4u);
  EXPECT_FLOAT_EQ(row[0], 0.f);
  EXPECT_FLOAT_EQ(row[3], 3.f);
}

// A mutable sub-view writes through to the backing storage.
TEST_F(Tensor3F32Fixture, AtMutableWritesThrough) {
  auto row = MutableView().At(1).At(2).As<float>();
  ASSERT_EQ(row.size(), 4u);
  row[1] = 999.f;

  // Element (1, 2, 1) has flat index 21.
  EXPECT_FLOAT_EQ(data_[21], 999.f);
  // Neighbors are untouched.
  EXPECT_FLOAT_EQ(data_[20], 20.f);
  EXPECT_FLOAT_EQ(data_[22], 22.f);
}

// Mutability is preserved through At (result stays a mutable view).
TEST_F(Tensor3F32Fixture, AtPreservesMutability) {
  static_assert(std::is_same_v<decltype(MutableView().At(0)), MutableMatrixView>);
  static_assert(std::is_same_v<decltype(MutableView().At(0).At(0)),
                               MutableVectorView>);
  static_assert(std::is_same_v<decltype(View().At(0)), MatrixView>);
}

// Q8_0 packs 32 elements per 34-byte block (2-byte f16 scale + 32 int8). Each
// block is filled with a marker byte equal to its global block index, so a
// sub-view's data pointer can be identified by the byte value it lands on.
//
// Buffer is 8 blocks laid out as a [2, 2, 64] tensor (2 matrices x 2 rows x 64
// elements = 2 blocks per row). C-order byte strides are {136, 68, 34}.
//
//   matrix 0, row 0 -> blocks 0,1   (byte 0)
//   matrix 0, row 1 -> blocks 2,3   (byte 68)
//   matrix 1, row 0 -> blocks 4,5   (byte 136)
//   matrix 1, row 1 -> blocks 6,7   (byte 204)
class Tensor3Q8Fixture : public ::testing::Test {
 protected:
  static constexpr int kBlockBytes = 34;  // sizeof(Float16) + 32 * sizeof(int8)
  static constexpr int kNumBlocks = 8;

  void SetUp() override {
    data_.resize(kNumBlocks * kBlockBytes);
    for (int b = 0; b < kNumBlocks; ++b) {
      for (int i = 0; i < kBlockBytes; ++i) {
        data_[b * kBlockBytes + i] = static_cast<std::byte>(b);
      }
    }
  }

  const std::byte* base() const { return data_.data(); }
  std::byte* mutable_base() { return data_.data(); }

  // A [2, 64] view over the first 4 blocks.
  MatrixView Matrix() const {
    return MatrixView::Create(DType::Q8_0, base(), 2, 64);
  }
  Tensor3View Tensor3() const {
    return Tensor3View::Create(DType::Q8_0, base(), 2, 2, 64);
  }
  MutableTensor3View MutableTensor3() {
    return MutableTensor3View::Create(DType::Q8_0, mutable_base(), 2, 2, 64);
  }

  std::vector<std::byte> data_;
};

// At on a Q8_0 matrix yields the n-th row with the block dtype/stride preserved.
TEST_F(Tensor3Q8Fixture, At2DMetadata) {
  auto row = Matrix().At(1);

  EXPECT_EQ(row.dtype, DType::Q8_0);
  // Offset = stride[0] * index = 68 * 1 bytes = block 2.
  EXPECT_EQ(row.data, base() + 68);
  EXPECT_EQ(row.shape[0], 64);
  EXPECT_EQ(row.stride[0], 34);  // bytes per block, carried through
}

// The row spans exactly its two blocks; markers identify which blocks.
TEST_F(Tensor3Q8Fixture, At2DLandsOnCorrectBlocks) {
  auto row0 = Matrix().At(0).As<const std::byte>();
  ASSERT_EQ(row0.size(), 68u);              // 64 elems / 32 per block * 34 bytes
  EXPECT_EQ(row0[0], static_cast<std::byte>(0));   // block 0
  EXPECT_EQ(row0[34], static_cast<std::byte>(1));  // block 1

  auto row1 = Matrix().At(1).As<const std::byte>();
  ASSERT_EQ(row1.size(), 68u);
  EXPECT_EQ(row1[0], static_cast<std::byte>(2));   // block 2
  EXPECT_EQ(row1[34], static_cast<std::byte>(3));  // block 3
}

// Chained At on a Q8_0 rank-3 view resolves to the right block, dtype intact.
TEST_F(Tensor3Q8Fixture, At3DChainedLandsOnCorrectBlock) {
  auto row = Tensor3().At(1).At(1);  // matrix 1, row 1 -> block 6 @ byte 204

  EXPECT_EQ(row.dtype, DType::Q8_0);
  EXPECT_EQ(row.data, base() + 204);
  EXPECT_EQ(row.shape[0], 64);
  EXPECT_EQ(row.stride[0], 34);

  auto bytes = row.As<const std::byte>();
  ASSERT_EQ(bytes.size(), 68u);
  EXPECT_EQ(bytes[0], static_cast<std::byte>(6));   // block 6
  EXPECT_EQ(bytes[34], static_cast<std::byte>(7));  // block 7
}

// The intermediate matrix from a rank-3 Q8_0 view also carries the dtype.
TEST_F(Tensor3Q8Fixture, At3DIntermediateMetadata) {
  auto m = Tensor3().At(1);  // matrix 1 -> blocks 4..7 @ byte 136

  EXPECT_EQ(m.dtype, DType::Q8_0);
  EXPECT_EQ(m.data, base() + 136);
  EXPECT_EQ(m.shape[0], 2);
  EXPECT_EQ(m.shape[1], 64);
  EXPECT_EQ(m.stride[0], 68);
  EXPECT_EQ(m.stride[1], 34);
}

// A mutable Q8_0 sub-view writes through to the backing block.
TEST_F(Tensor3Q8Fixture, AtMutableWritesThrough) {
  auto bytes = MutableTensor3().At(1).At(1).As<std::byte>();  // block 6 @ byte 204
  ASSERT_EQ(bytes.size(), 68u);
  bytes[5] = static_cast<std::byte>(0xAB);

  EXPECT_EQ(data_[204 + 5], static_cast<std::byte>(0xAB));
  // A neighboring block is untouched.
  EXPECT_EQ(data_[204 + 34], static_cast<std::byte>(7));
}

// ── TensorView::Create ────────────────────────────────────────────────────────
//
// Create is a static factory: the enclosing view type fixes Rank + mutability,
// the size arguments give the shape, and it fills contiguous (row-major),
// block-aware strides for the given dtype.

TEST(TensorViewCreate, VectorF32) {
  std::vector<float> buf(8);
  auto* data = reinterpret_cast<std::byte*>(buf.data());

  auto v = MutableVectorView::Create(DType::F32, data, 8);

  EXPECT_EQ(v.dtype, DType::F32);
  EXPECT_EQ(v.data, data);
  EXPECT_EQ(v.shape[0], 8);
  EXPECT_EQ(v.stride[0], 4);        // bytes per F32 block
  EXPECT_EQ(v.TotalBytes(), 32);    // 8 * 4
}

TEST(TensorViewCreate, MatrixF32) {
  std::vector<float> buf(3 * 4);
  auto* data = reinterpret_cast<std::byte*>(buf.data());

  auto m = MutableMatrixView::Create(DType::F32, data, 3, 4);

  EXPECT_EQ(m.dtype, DType::F32);
  EXPECT_EQ(m.shape[0], 3);
  EXPECT_EQ(m.shape[1], 4);
  EXPECT_EQ(m.stride[0], 16);       // 4 cols * 4 bytes/block
  EXPECT_EQ(m.stride[1], 4);
  EXPECT_EQ(m.TotalBytes(), 48);
}

TEST(TensorViewCreate, Tensor3F32Strides) {
  std::vector<float> buf(2 * 3 * 4);
  auto* data = reinterpret_cast<std::byte*>(buf.data());

  auto t = MutableTensor3View::Create(DType::F32, data, 2, 3, 4);

  EXPECT_EQ(t.shape[0], 2);
  EXPECT_EQ(t.shape[1], 3);
  EXPECT_EQ(t.shape[2], 4);
  EXPECT_EQ(t.stride[0], 48);
  EXPECT_EQ(t.stride[1], 16);
  EXPECT_EQ(t.stride[2], 4);
}

TEST(TensorViewCreate, MatrixQ8BlockAwareStrides) {
  std::vector<std::byte> buf(2 * (64 / 32) * 34);  // 2 rows, 2 blocks each

  auto m = MutableMatrixView::Create(DType::Q8_0, buf.data(), 2, 64);

  EXPECT_EQ(m.dtype, DType::Q8_0);
  EXPECT_EQ(m.shape[0], 2);
  EXPECT_EQ(m.shape[1], 64);
  EXPECT_EQ(m.stride[0], 68);       // (64 / 32 blocks) * 34 bytes
  EXPECT_EQ(m.stride[1], 34);       // bytes per Q8_0 block
}

TEST(TensorViewCreate, Tensor3Q8BlockAwareStrides) {
  std::vector<std::byte> buf(2 * 2 * (64 / 32) * 34);

  auto t = MutableTensor3View::Create(DType::Q8_0, buf.data(), 2, 2, 64);

  EXPECT_EQ(t.stride[0], 136);      // shape[1] * stride[1]
  EXPECT_EQ(t.stride[1], 68);       // (64 / 32) * 34
  EXPECT_EQ(t.stride[2], 34);       // bytes per block
}

// Create works on the const specialization too (Byte* is const std::byte*).
TEST(TensorViewCreate, ConstView) {
  std::vector<float> buf(4);
  const std::byte* data = reinterpret_cast<const std::byte*>(buf.data());

  auto v = VectorView::Create(DType::F32, data, 4);

  static_assert(std::is_same_v<decltype(v), VectorView>);
  EXPECT_EQ(v.data, data);
  EXPECT_EQ(v.shape[0], 4);
  EXPECT_EQ(v.stride[0], 4);
}

// The enclosing type's mutability flows into the created view.
TEST(TensorViewCreate, MutabilityPropagates) {
  std::vector<float> buf(4);
  auto* mdata = reinterpret_cast<std::byte*>(buf.data());
  const std::byte* cdata = reinterpret_cast<const std::byte*>(buf.data());

  static_assert(std::is_same_v<
      decltype(MutableMatrixView::Create(DType::F32, mdata, 2, 2)),
      MutableMatrixView>);
  static_assert(std::is_same_v<
      decltype(MatrixView::Create(DType::F32, cdata, 2, 2)), MatrixView>);
}

// Integration: strides from Create navigate correctly through At + As.
TEST(TensorViewCreate, NavigatesWithAtAndAs) {
  std::vector<float> buf(2 * 3 * 4);
  for (int i = 0; i < static_cast<int>(buf.size()); ++i) {
    buf[i] = static_cast<float>(i);
  }
  auto* data = reinterpret_cast<std::byte*>(buf.data());

  auto t = MutableTensor3View::Create(DType::F32, data, 2, 3, 4);
  auto row = t.At(1).At(2).As<float>();  // element (1, 2, :) = flat 20..23

  ASSERT_EQ(row.size(), 4u);
  EXPECT_FLOAT_EQ(row[0], 20.f);
  EXPECT_FLOAT_EQ(row[1], 21.f);
  EXPECT_FLOAT_EQ(row[2], 22.f);
  EXPECT_FLOAT_EQ(row[3], 23.f);
}

// ── Mutable -> immutable conversion ───────────────────────────────────────────
//
// A mutable view converts implicitly to the immutable view of the same rank
// (the span<T> -> span<const T> direction). The reverse never compiles.

// A [4, 3] F32 matrix with element (i, j) = 10*i + j.
class ConversionFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    data_.resize(4 * 3);
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 3; ++j) data_[i * 3 + j] = 10.f * i + j;
    }
  }

  MutableMatrixView Mutable() {
    return MutableMatrixView::Create(
        DType::F32, reinterpret_cast<std::byte*>(data_.data()), 4, 3);
  }

  std::vector<float> data_;
};

// Something that only accepts an immutable view, like every kernel input.
int64_t RowsOf(MatrixView m) { return m.shape[0]; }

TEST_F(ConversionFixture, MutableConvertsToImmutableImplicitly) {
  MutableMatrixView mut = Mutable();
  MatrixView imm = mut;  // copy-initialization, no cast

  EXPECT_EQ(imm.dtype, mut.dtype);
  EXPECT_EQ(imm.data, mut.data);  // same memory, not a copy of the data
  EXPECT_EQ(imm.shape, mut.shape);
  EXPECT_EQ(imm.stride, mut.stride);
  EXPECT_EQ(RowsOf(mut), 4);  // passes straight into a MatrixView parameter
}

TEST_F(ConversionFixture, ImmutableNeverConvertsToMutable) {
  static_assert(!std::is_convertible_v<MatrixView, MutableMatrixView>);
  static_assert(!std::is_constructible_v<MutableMatrixView, MatrixView>);
  static_assert(!std::is_convertible_v<VectorView, MutableVectorView>);
  static_assert(!std::is_convertible_v<Tensor3View, MutableTensor3View>);
  // Same-mutability copies still work in both flavours.
  static_assert(std::is_copy_constructible_v<MatrixView>);
  static_assert(std::is_copy_constructible_v<MutableMatrixView>);
}

// The converted view reads the same bytes the mutable one writes.
TEST_F(ConversionFixture, ConvertedViewObservesWrites) {
  MutableMatrixView mut = Mutable();
  MatrixView imm = mut;

  mut.As<float>()[4] = 99.f;  // element (1, 1)

  EXPECT_FLOAT_EQ(imm.As<const float>()[4], 99.f);
}

// ── Slice / Top / Bottom ──────────────────────────────────────────────────────
//
// Slicing along the first dimension keeps the mutability of the source view;
// the result converts to immutable like any other mutable view.

TEST_F(ConversionFixture, SlicePreservesMutability) {
  MutableMatrixView mut = Mutable();
  const MatrixView imm = mut;

  static_assert(std::is_same_v<decltype(mut.Slice(0, 1)), MutableMatrixView>);
  static_assert(std::is_same_v<decltype(mut.Top(1)), MutableMatrixView>);
  static_assert(std::is_same_v<decltype(mut.Bottom(1)), MutableMatrixView>);
  static_assert(std::is_same_v<decltype(imm.Slice(0, 1)), MatrixView>);
  static_assert(std::is_same_v<decltype(imm.Top(1)), MatrixView>);
  static_assert(std::is_same_v<decltype(imm.Bottom(1)), MatrixView>);
}

TEST_F(ConversionFixture, SliceOffsetsAndShape) {
  MatrixView m = Mutable();
  MatrixView s = m.Slice(1, 2);  // rows 1..2

  EXPECT_EQ(s.data, m.data + m.stride[0]);
  EXPECT_EQ(s.shape[0], 2);
  EXPECT_EQ(s.shape[1], 3);  // inner dim untouched
  EXPECT_EQ(s.stride, m.stride);
  EXPECT_EQ(s.TotalBytes(), 2 * 3 * 4);

  std::span<const float> vals = s.As<const float>();
  ASSERT_EQ(vals.size(), 6u);
  EXPECT_FLOAT_EQ(vals[0], 10.f);  // (1, 0)
  EXPECT_FLOAT_EQ(vals[5], 22.f);  // (2, 2)
}

TEST_F(ConversionFixture, TopAndBottomAreSliceEndpoints) {
  MatrixView m = Mutable();

  MatrixView top = m.Top(2);
  EXPECT_EQ(top.data, m.data);
  EXPECT_EQ(top.shape[0], 2);

  MatrixView bottom = m.Bottom(1);
  EXPECT_EQ(bottom.data, m.data + 3 * m.stride[0]);
  EXPECT_EQ(bottom.shape[0], 1);
  EXPECT_FLOAT_EQ(bottom.As<const float>()[0], 30.f);  // (3, 0)
}

// A mutable slice writes through to the backing rows, and only those rows.
TEST_F(ConversionFixture, MutableSliceWritesThrough) {
  MutableMatrixView rows = Mutable().Slice(2, 1);  // row 2
  std::span<float> vals = rows.As<float>();
  ASSERT_EQ(vals.size(), 3u);
  vals[1] = -1.f;

  EXPECT_FLOAT_EQ(data_[2 * 3 + 1], -1.f);
  EXPECT_FLOAT_EQ(data_[1 * 3 + 1], 11.f);  // row above untouched
  EXPECT_FLOAT_EQ(data_[3 * 3 + 1], 31.f);  // row below untouched
}

// The everyday pattern from the blocks: slice a mutable buffer, hand the
// slice to a kernel as its output, then pass the same slice on as an
// immutable input with no explicit conversion.
TEST_F(ConversionFixture, MutableSliceFlowsIntoImmutableParameter) {
  MutableMatrixView out = Mutable().Top(2);
  EXPECT_EQ(RowsOf(out), 2);
  MatrixView next_input = out;
  EXPECT_EQ(next_input.data, out.data);
}

}  // namespace
}  // namespace tlm
