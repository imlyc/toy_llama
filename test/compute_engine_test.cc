#include "compute/compute_engine.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/dtype.h"
#include "tensor/tensor_view.h"

namespace tlm {
namespace {

VectorView AsVec(const std::vector<float>& v) {
  return VectorView::Create(DType::F32,
                            reinterpret_cast<const std::byte*>(v.data()),
                            static_cast<int64_t>(v.size()));
}

MutableVectorView AsMutableVec(std::vector<float>& v) {
  return MutableVectorView::Create(DType::F32,
                                   reinterpret_cast<std::byte*>(v.data()),
                                   static_cast<int64_t>(v.size()));
}

MatrixView AsMat(const std::vector<float>& v, int64_t rows, int64_t cols) {
  return MatrixView::Create(DType::F32,
                            reinterpret_cast<const std::byte*>(v.data()), rows,
                            cols);
}

// One token, one head: softmax over a single score is 1, so the output must be
// exactly the (only) V row, independent of q and k.
TEST(ComputeEngineAttnTest, SingleTokenCopiesV) {
  ComputeEngine engine;
  std::vector<float> q = {5.f, -3.f};
  std::vector<float> k = {1.f, 2.f};
  std::vector<float> v = {7.f, 9.f};
  std::vector<float> out(2, -1.f);

  engine.Attn(AsMutableVec(out), AsVec(q), AsMat(k, 1, 2), AsMat(v, 1, 2),
              /*head_count_q=*/1, /*head_count_kv=*/1);

  EXPECT_NEAR(out[0], 7.f, 1e-5);
  EXPECT_NEAR(out[1], 9.f, 1e-5);
}

// Identical K rows give identical scores, so the attention weights must be
// uniform (1/3 each) and the output is the mean of the V rows. Verifies the
// softmax normalizes regardless of q.
TEST(ComputeEngineAttnTest, EqualScoresAverageV) {
  ComputeEngine engine;
  std::vector<float> q = {0.7f, -0.2f};
  std::vector<float> k = {
      1.f, 1.f,  // t = 0
      1.f, 1.f,  // t = 1
      1.f, 1.f,  // t = 2
  };
  std::vector<float> v = {
      3.f, 0.f,  // t = 0
      6.f, 9.f,  // t = 1
      0.f, 3.f,  // t = 2
  };
  std::vector<float> out(2, -1.f);

  engine.Attn(AsMutableVec(out), AsVec(q), AsMat(k, 3, 2), AsMat(v, 3, 2),
              /*head_count_q=*/1, /*head_count_kv=*/1);

  EXPECT_NEAR(out[0], 3.f, 1e-4);  // (3 + 6 + 0) / 3
  EXPECT_NEAR(out[1], 4.f, 1e-4);  // (0 + 9 + 3) / 3
}

// 4 query heads grouped onto 2 kv heads (q_per_kv = 2), head_dim 2, seq 2.
//
// The constant a = ln(3) * sqrt(2) is chosen so that a raw score difference of
// `a` becomes ln(3) after the 1/sqrt(dk) scale, making the softmax weights
// exactly (0.25, 0.75). Each head uses a different q so that head grouping,
// per-head q/k/v slicing, the scale, and the softmax all affect the result:
//
//   head 0 (kv 0): q=(a,0), kv0 keys (0,0),(1,0) -> scores (0,a) -> p=(1/4,3/4)
//   head 1 (kv 0): q=(0,0)                       -> scores (0,0) -> p=(1/2,1/2)
//   head 2 (kv 1): q=(a,0), kv1 keys (1,0),(0,1) -> scores (a,0) -> p=(3/4,1/4)
//   head 3 (kv 1): q=(0,a)                       -> scores (0,a) -> p=(1/4,3/4)
TEST(ComputeEngineAttnTest, GroupedHeadsHandComputed) {
  ComputeEngine engine;
  const float a = std::log(3.f) * std::sqrt(2.f);

  std::vector<float> q = {
      a,   0.f,  // head 0
      0.f, 0.f,  // head 1
      a,   0.f,  // head 2
      0.f, a,    // head 3
  };
  // Rows are [kv0_d0, kv0_d1, kv1_d0, kv1_d1].
  std::vector<float> k = {
      0.f, 0.f, 1.f, 0.f,  // t = 0
      1.f, 0.f, 0.f, 1.f,  // t = 1
  };
  std::vector<float> v = {
      1.f, 2.f, 10.f, 20.f,  // t = 0
      3.f, 6.f, 30.f, 60.f,  // t = 1
  };
  std::vector<float> out(8, -1.f);

  engine.Attn(AsMutableVec(out), AsVec(q), AsMat(k, 2, 4), AsMat(v, 2, 4),
              /*head_count_q=*/4, /*head_count_kv=*/2);

  const std::vector<float> expected = {
      2.5f, 5.f,   // head 0: 0.25*(1,2)   + 0.75*(3,6)
      2.f,  4.f,   // head 1: 0.5 *(1,2)   + 0.5 *(3,6)
      15.f, 30.f,  // head 2: 0.75*(10,20) + 0.25*(30,60)
      25.f, 50.f,  // head 3: 0.25*(10,20) + 0.75*(30,60)
  };
  for (int i = 0; i < 8; ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-4) << "out[" << i << "]";
  }
}

// K and V passed as Top() slices of a taller cache: only the first `len` rows
// may influence the result — mirrors how the GQA block slices its KV cache.
TEST(ComputeEngineAttnTest, TopSliceIgnoresRowsBeyondLength) {
  ComputeEngine engine;
  std::vector<float> q = {0.3f, 0.9f};
  // 4-row caches; rows 2..3 hold garbage that must not leak into the result.
  std::vector<float> k = {
      1.f, 1.f,      // t = 0
      1.f, 1.f,      // t = 1
      999.f, -99.f,  // beyond len
      -99.f, 999.f,  // beyond len
  };
  std::vector<float> v = {
      2.f, 4.f,      // t = 0
      6.f, 8.f,      // t = 1
      1e6f, 1e6f,    // beyond len
      -1e6f, -1e6f,  // beyond len
  };
  std::vector<float> out(2, -1.f);

  MatrixView k_top = AsMat(k, 4, 2).Top(2);
  MatrixView v_top = AsMat(v, 4, 2).Top(2);
  engine.Attn(AsMutableVec(out), AsVec(q), k_top, v_top,
              /*head_count_q=*/1, /*head_count_kv=*/1);

  // Equal keys in the valid prefix -> uniform weights -> mean of first 2 rows.
  EXPECT_NEAR(out[0], 4.f, 1e-4);
  EXPECT_NEAR(out[1], 6.f, 1e-4);
}

// Add must OVERWRITE out with lhs + rhs. out is pre-filled with a sentinel to
// catch implementations that accumulate into existing contents.
TEST(ComputeEngineAddTest, OverwritesOut) {
  ComputeEngine engine;
  std::vector<float> lhs = {1.f, -2.f, 0.5f, 0.f};
  std::vector<float> rhs = {10.f, 20.f, -0.5f, 0.f};
  std::vector<float> out = {99.f, 99.f, 99.f, 99.f};  // stale garbage

  engine.Add(AsMutableVec(out), AsVec(lhs), AsVec(rhs));

  EXPECT_FLOAT_EQ(out[0], 11.f);
  EXPECT_FLOAT_EQ(out[1], 18.f);
  EXPECT_FLOAT_EQ(out[2], 0.f);
  EXPECT_FLOAT_EQ(out[3], 0.f);
}

// Calling Add twice into the same out must give the same result both times
// (no accumulation across calls) — mirrors how block buffers are reused
// across tokens.
TEST(ComputeEngineAddTest, RepeatedCallsAreIdempotent) {
  ComputeEngine engine;
  std::vector<float> lhs = {1.f, 2.f};
  std::vector<float> rhs = {3.f, 4.f};
  std::vector<float> out(2, 0.f);

  engine.Add(AsMutableVec(out), AsVec(lhs), AsVec(rhs));
  engine.Add(AsMutableVec(out), AsVec(lhs), AsVec(rhs));

  EXPECT_FLOAT_EQ(out[0], 4.f);
  EXPECT_FLOAT_EQ(out[1], 6.f);
}

// In-place add: out aliases lhs (elementwise op, so aliasing is safe by
// contract). out = out + rhs.
TEST(ComputeEngineAddTest, InPlaceLhsAlias) {
  ComputeEngine engine;
  std::vector<float> lhs = {1.f, 2.f, 3.f};
  std::vector<float> rhs = {10.f, 20.f, 30.f};

  engine.Add(AsMutableVec(lhs), AsVec(lhs), AsVec(rhs));

  EXPECT_FLOAT_EQ(lhs[0], 11.f);
  EXPECT_FLOAT_EQ(lhs[1], 22.f);
  EXPECT_FLOAT_EQ(lhs[2], 33.f);
}

}  // namespace
}  // namespace tlm
