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

MutableMatrixView AsMutableMat(std::vector<float>& v, int64_t rows,
                               int64_t cols) {
  return MutableMatrixView::Create(DType::F32,
                                   reinterpret_cast<std::byte*>(v.data()),
                                   rows, cols);
}

// Single-row [1, n] matrix views: the T = 1 (decode) case of every matrix
// kernel. Lets the original vector-level cases run unchanged as regressions.
MatrixView AsRow(const std::vector<float>& v) {
  return AsMat(v, 1, static_cast<int64_t>(v.size()));
}

MutableMatrixView AsMutableRow(std::vector<float>& v) {
  return AsMutableMat(v, 1, static_cast<int64_t>(v.size()));
}

MatrixView AsQ8Mat(const std::vector<BlockQ8_0>& blocks, int64_t rows,
                   int64_t cols) {
  return MatrixView::Create(DType::Q8_0,
                            reinterpret_cast<const std::byte*>(blocks.data()),
                            rows, cols);
}

// ── Attn ──────────────────────────────────────────────────────────────────────
// q is [T, head_count_q * head_dim]; k, v are [L, head_count_kv * head_dim]
// with L >= T. The last T rows of k/v correspond to the T query rows, so
// query row t may attend to keys [0, L - T + t] only (causal mask).

// One token, one head: softmax over a single score is 1, so the output must be
// exactly the (only) V row, independent of q and k.
TEST(ComputeEngineAttnTest, SingleTokenCopiesV) {
  ComputeEngine engine;
  std::vector<float> q = {5.f, -3.f};
  std::vector<float> k = {1.f, 2.f};
  std::vector<float> v = {7.f, 9.f};
  std::vector<float> out(2, -1.f);

  engine.Attn(AsMutableRow(out), AsRow(q), AsMat(k, 1, 2), AsMat(v, 1, 2),
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

  engine.Attn(AsMutableRow(out), AsRow(q), AsMat(k, 3, 2), AsMat(v, 3, 2),
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

  engine.Attn(AsMutableRow(out), AsRow(q), AsMat(k, 2, 4), AsMat(v, 2, 4),
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
  engine.Attn(AsMutableRow(out), AsRow(q), k_top, v_top,
              /*head_count_q=*/1, /*head_count_kv=*/1);

  // Equal keys in the valid prefix -> uniform weights -> mean of first 2 rows.
  EXPECT_NEAR(out[0], 4.f, 1e-4);
  EXPECT_NEAR(out[1], 6.f, 1e-4);
}

// Two queries over their own two keys (no prior context). Identical keys make
// every visible subset uniform, so the mask alone decides the output: row 0
// may see only key 0, row 1 sees both. A missing mask would average both rows
// into row 0 as well.
TEST(ComputeEngineAttnTest, CausalMaskHidesFutureKeys) {
  ComputeEngine engine;
  std::vector<float> q = {
      0.5f, -0.5f,  // query 0
      2.f,  1.f,    // query 1
  };
  std::vector<float> k = {
      1.f, 1.f,  // key 0
      1.f, 1.f,  // key 1
  };
  std::vector<float> v = {
      2.f, 4.f,  // t = 0
      6.f, 8.f,  // t = 1
  };
  std::vector<float> out(4, -1.f);

  engine.Attn(AsMutableMat(out, 2, 2), AsMat(q, 2, 2), AsMat(k, 2, 2),
              AsMat(v, 2, 2), /*head_count_q=*/1, /*head_count_kv=*/1);

  EXPECT_NEAR(out[0], 2.f, 1e-4);  // row 0: only v0
  EXPECT_NEAR(out[1], 4.f, 1e-4);
  EXPECT_NEAR(out[2], 4.f, 1e-4);  // row 1: mean(v0, v1)
  EXPECT_NEAR(out[3], 6.f, 1e-4);
}

// Prefill on top of existing context: L = 4 cached keys, T = 2 new queries.
// The new queries are the LAST two positions, so row 0 sees keys 0..2 and
// row 1 sees keys 0..3. Pins the mask offset (L - T) for the non-empty-cache
// case.
TEST(ComputeEngineAttnTest, CausalMaskWithPriorContext) {
  ComputeEngine engine;
  std::vector<float> q = {
      1.f, 0.f,  // query at position 2
      0.f, 1.f,  // query at position 3
  };
  std::vector<float> k(8, 1.f);  // 4 identical keys -> uniform over prefix
  std::vector<float> v = {
      0.f,  0.f,   // t = 0
      4.f,  4.f,   // t = 1
      8.f,  8.f,   // t = 2
      12.f, 12.f,  // t = 3
  };
  std::vector<float> out(4, -1.f);

  engine.Attn(AsMutableMat(out, 2, 2), AsMat(q, 2, 2), AsMat(k, 4, 2),
              AsMat(v, 4, 2), /*head_count_q=*/1, /*head_count_kv=*/1);

  EXPECT_NEAR(out[0], 4.f, 1e-4);  // row 0: mean(v0, v1, v2)
  EXPECT_NEAR(out[1], 4.f, 1e-4);
  EXPECT_NEAR(out[2], 6.f, 1e-4);  // row 1: mean(v0 .. v3)
  EXPECT_NEAR(out[3], 6.f, 1e-4);
}

// The prefill contract: a batched call over T queries must equal T serial
// single-query calls, each attending over Top(t + 1). Uses GQA (2 q heads on
// 1 kv head) and non-degenerate values so the scores, scale, mask, softmax and
// per-head slicing all participate.
TEST(ComputeEngineAttnTest, BatchedMatchesSerial) {
  ComputeEngine engine;
  const int64_t T = 3, q_cols = 4, kv_cols = 2;
  std::vector<float> q = {
      0.9f,  -0.4f, 0.2f,  1.1f,   // t = 0: head 0 | head 1
      -1.2f, 0.3f,  0.8f,  -0.7f,  // t = 1
      0.5f,  0.5f,  -1.5f, 0.1f,   // t = 2
  };
  std::vector<float> k = {
      0.3f, -1.f,   // t = 0
      1.2f, 0.4f,   // t = 1
      -0.6f, 0.9f,  // t = 2
  };
  std::vector<float> v = {
      1.f,  2.f,   // t = 0
      -3.f, 0.5f,  // t = 1
      4.f,  -1.f,  // t = 2
  };

  std::vector<float> batched(T * q_cols, -1.f);
  engine.Attn(AsMutableMat(batched, T, q_cols), AsMat(q, T, q_cols),
              AsMat(k, T, kv_cols), AsMat(v, T, kv_cols),
              /*head_count_q=*/2, /*head_count_kv=*/1);

  for (int64_t t = 0; t < T; ++t) {
    std::vector<float> serial(q_cols, -1.f);
    engine.Attn(AsMutableRow(serial), AsMat(q, T, q_cols).Slice(t, 1),
                AsMat(k, T, kv_cols).Top(t + 1),
                AsMat(v, T, kv_cols).Top(t + 1),
                /*head_count_q=*/2, /*head_count_kv=*/1);
    for (int64_t c = 0; c < q_cols; ++c) {
      EXPECT_NEAR(batched[t * q_cols + c], serial[c], 1e-5)
          << "t=" << t << " c=" << c;
    }
  }
}

// ── Add ───────────────────────────────────────────────────────────────────────

// Add must OVERWRITE out with lhs + rhs. out is pre-filled with a sentinel to
// catch implementations that accumulate into existing contents.
TEST(ComputeEngineAddTest, OverwritesOut) {
  ComputeEngine engine;
  std::vector<float> lhs = {1.f, -2.f, 0.5f, 0.f};
  std::vector<float> rhs = {10.f, 20.f, -0.5f, 0.f};
  std::vector<float> out = {99.f, 99.f, 99.f, 99.f};  // stale garbage

  engine.Add(AsMutableRow(out), AsRow(lhs), AsRow(rhs));

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

  engine.Add(AsMutableRow(out), AsRow(lhs), AsRow(rhs));
  engine.Add(AsMutableRow(out), AsRow(lhs), AsRow(rhs));

  EXPECT_FLOAT_EQ(out[0], 4.f);
  EXPECT_FLOAT_EQ(out[1], 6.f);
}

// In-place add: out aliases lhs (elementwise op, so aliasing is safe by
// contract). out = out + rhs.
TEST(ComputeEngineAddTest, InPlaceLhsAlias) {
  ComputeEngine engine;
  std::vector<float> lhs = {1.f, 2.f, 3.f};
  std::vector<float> rhs = {10.f, 20.f, 30.f};

  engine.Add(AsMutableRow(lhs), AsRow(lhs), AsRow(rhs));

  EXPECT_FLOAT_EQ(lhs[0], 11.f);
  EXPECT_FLOAT_EQ(lhs[1], 22.f);
  EXPECT_FLOAT_EQ(lhs[2], 33.f);
}

// Multi-row operands: every element is added with its own counterpart; rows
// must not bleed into each other.
TEST(ComputeEngineAddTest, MultiRowElementwise) {
  ComputeEngine engine;
  std::vector<float> lhs = {1.f, 2.f,
                            3.f, 4.f};
  std::vector<float> rhs = {10.f, 20.f,
                            30.f, 40.f};
  std::vector<float> out(4, 99.f);

  engine.Add(AsMutableMat(out, 2, 2), AsMat(lhs, 2, 2), AsMat(rhs, 2, 2));

  EXPECT_FLOAT_EQ(out[0], 11.f);
  EXPECT_FLOAT_EQ(out[1], 22.f);
  EXPECT_FLOAT_EQ(out[2], 33.f);
  EXPECT_FLOAT_EQ(out[3], 44.f);
}

// ── MatMul ────────────────────────────────────────────────────────────────────
// Convention: out = mat @ vec with mat shaped [out, in], rows contiguous
// (GGUF weight layout; Q8_0 blocks run along the summed `in` dimension).

TEST(ComputeEngineMatMulTest, F32HandComputed) {
  ComputeEngine engine;
  // W = [out=2, in=3], rows contiguous; out = W @ x.
  std::vector<float> w = {1.f, 2.f, 3.f,
                          4.f, 5.f, 6.f};
  std::vector<float> x = {1.f, 0.5f, -1.f};
  std::vector<float> out = {99.f, 99.f};  // sentinel: MatMul must overwrite

  engine.MatMul(AsMutableVec(out), AsMat(w, 2, 3), AsVec(x));

  EXPECT_FLOAT_EQ(out[0], -1.f);   // 1 + 1 - 3
  EXPECT_FLOAT_EQ(out[1], 0.5f);   // 4 + 2.5 - 6
}

// Q8 mat is [out=2, in=32]: one block per row, blocks along the summed `in`
// dimension. data[i] = i with x alternating 0/1 makes the dot order-sensitive.
TEST(ComputeEngineMatMulTest, Q8SingleBlockPerRow) {
  ComputeEngine engine;
  std::vector<BlockQ8_0> blocks(2);
  blocks[0].scale = 0.5f;
  blocks[1].scale = 2.f;
  for (int i = 0; i < 32; ++i) {
    blocks[0].data[i] = static_cast<int8_t>(i);
    blocks[1].data[i] = 1;
  }
  std::vector<float> x(32);
  for (int i = 0; i < 32; ++i) x[i] = static_cast<float>(i % 2);  // odd picks
  std::vector<float> out = {99.f, 99.f};

  engine.MatMul(AsMutableVec(out), AsQ8Mat(blocks, 2, 32), AsVec(x));

  // Row 0: 0.5 * (sum of odd i in [0,32)) = 0.5 * 256.
  EXPECT_FLOAT_EQ(out[0], 128.f);
  // Row 1: 2 * (number of odd i) = 2 * 16.
  EXPECT_FLOAT_EQ(out[1], 32.f);
}

// Two blocks per row (in = 64) with different scales: catches block-count and
// block-indexing mistakes.
TEST(ComputeEngineMatMulTest, Q8MultiBlockRow) {
  ComputeEngine engine;
  std::vector<BlockQ8_0> blocks(2);  // one row: out = 1, in = 64
  blocks[0].scale = 1.f;
  blocks[1].scale = 2.f;
  for (int i = 0; i < 32; ++i) {
    blocks[0].data[i] = 1;
    blocks[1].data[i] = 1;
  }
  std::vector<float> x(64, 1.f);
  std::vector<float> out = {99.f};

  engine.MatMul(AsMutableVec(out), AsQ8Mat(blocks, 1, 64), AsVec(x));

  EXPECT_FLOAT_EQ(out[0], 96.f);  // 1*32 + 2*32
}

// With scale = 1 and small integer weights, the Q8 path must agree exactly
// with the F32 path on the same logical [out, in] matrix.
TEST(ComputeEngineMatMulTest, Q8MatchesF32) {
  ComputeEngine engine;
  const int64_t rows = 3, cols = 32;  // out = 3, in = 32
  std::vector<BlockQ8_0> blocks(rows);
  std::vector<float> w_f32(rows * cols);
  for (int64_t r = 0; r < rows; ++r) {
    blocks[r].scale = 1.f;
    for (int64_t i = 0; i < cols; ++i) {
      const int8_t q = static_cast<int8_t>((r * 7 + i * 3) % 21 - 10);
      blocks[r].data[i] = q;
      w_f32[r * cols + i] = static_cast<float>(q);
    }
  }
  std::vector<float> x(cols);
  for (int64_t i = 0; i < cols; ++i) x[i] = 0.25f * static_cast<float>(i - 16);

  std::vector<float> out_q8(rows, 99.f);
  std::vector<float> out_f32(rows, -99.f);
  engine.MatMul(AsMutableVec(out_q8), AsQ8Mat(blocks, rows, cols), AsVec(x));
  engine.MatMul(AsMutableVec(out_f32), AsMat(w_f32, rows, cols), AsVec(x));

  for (int64_t r = 0; r < rows; ++r) {
    EXPECT_FLOAT_EQ(out_q8[r], out_f32[r]) << "row " << r;
  }
}

// ── MatMulT ───────────────────────────────────────────────────────────────────
// Batched projection: out[T, out_dim] = A[T, in] @ W^T with W = [out_dim, in]
// in GGUF layout. Row t of out is W @ A[t], i.e. MatMul applied per token.

// A = [T=2, in=3], W = [out=2, in=3]. Each output row is W @ (that input row).
TEST(ComputeEngineMatMulTTest, F32HandComputed) {
  ComputeEngine engine;
  std::vector<float> a = {1.f, 0.5f, -1.f,
                          2.f, 1.f,  0.f};
  std::vector<float> w = {1.f, 2.f, 3.f,
                          4.f, 5.f, 6.f};
  std::vector<float> out(4, 99.f);

  engine.MatMulT(AsMutableMat(out, 2, 2), AsMat(a, 2, 3), AsMat(w, 2, 3));

  EXPECT_FLOAT_EQ(out[0], -1.f);   // row 0 . w0 = 1 + 1 - 3
  EXPECT_FLOAT_EQ(out[1], 0.5f);   // row 0 . w1 = 4 + 2.5 - 6
  EXPECT_FLOAT_EQ(out[2], 4.f);    // row 1 . w0 = 2 + 2 + 0
  EXPECT_FLOAT_EQ(out[3], 13.f);   // row 1 . w1 = 8 + 5 + 0
}

// T = 1 is the decode path: MatMulT on a [1, in] input must equal MatMul on
// the same vector. F32 weights.
TEST(ComputeEngineMatMulTTest, SingleRowMatchesMatMulF32) {
  ComputeEngine engine;
  std::vector<float> w = {1.f, 2.f, 3.f,
                          4.f, 5.f, 6.f,
                          -1.f, 0.f, 2.f};
  std::vector<float> x = {0.5f, -2.f, 1.5f};

  std::vector<float> mm(3, 99.f), mmt(3, -99.f);
  engine.MatMul(AsMutableVec(mm), AsMat(w, 3, 3), AsVec(x));
  engine.MatMulT(AsMutableRow(mmt), AsRow(x), AsMat(w, 3, 3));

  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(mmt[i], mm[i]) << "i=" << i;
  }
}

// Same for Q8_0 weights, with non-unit scales so the scale is exercised. The
// two kernels accumulate in a different order, hence NEAR not FLOAT_EQ.
TEST(ComputeEngineMatMulTTest, SingleRowMatchesMatMulQ8) {
  ComputeEngine engine;
  const int64_t rows = 3, cols = 64;  // out = 3, in = 64 (2 blocks per row)
  std::vector<BlockQ8_0> blocks(rows * 2);
  for (int64_t b = 0; b < rows * 2; ++b) {
    blocks[b].scale = 0.5f + 0.25f * static_cast<float>(b);
    for (int i = 0; i < 32; ++i) {
      blocks[b].data[i] = static_cast<int8_t>((b * 5 + i * 3) % 23 - 11);
    }
  }
  std::vector<float> x(cols);
  for (int64_t i = 0; i < cols; ++i) x[i] = 0.125f * static_cast<float>(i - 30);

  std::vector<float> mm(rows, 99.f), mmt(rows, -99.f);
  engine.MatMul(AsMutableVec(mm), AsQ8Mat(blocks, rows, cols), AsVec(x));
  engine.MatMulT(AsMutableRow(mmt), AsRow(x), AsQ8Mat(blocks, rows, cols));

  for (int64_t r = 0; r < rows; ++r) {
    EXPECT_NEAR(mmt[r], mm[r], 1e-3) << "row " << r;
  }
}

// T = 3 tokens, out_dim = 2, in = 64 (2 blocks per weight row), with every
// block distinct. The activation columns a block multiplies must be selected
// by the BLOCK index, not the weight-row index: with out_dim < in, indexing
// by row would read the wrong (or out-of-range) columns.
//
//   W row 0: block 0 = 1s (scale 1), block 1 = 2s (scale 1)
//   W row 1: block 0 = 1s (scale 2), block 1 = 0s
//   A row t: first 32 cols = (t+1), last 32 cols = 10(t+1)
//   out[t][0] = 32(t+1) + 2*32*10(t+1) = 672(t+1)
//   out[t][1] = 2*32(t+1)              = 64(t+1)
TEST(ComputeEngineMatMulTTest, Q8MultiRowDistinctBlocks) {
  ComputeEngine engine;
  const int64_t T = 3, in = 64, out_dim = 2;
  std::vector<BlockQ8_0> blocks(4);
  blocks[0].scale = 1.f;
  blocks[1].scale = 1.f;
  blocks[2].scale = 2.f;
  blocks[3].scale = 1.f;
  for (int i = 0; i < 32; ++i) {
    blocks[0].data[i] = 1;
    blocks[1].data[i] = 2;
    blocks[2].data[i] = 1;
    blocks[3].data[i] = 0;
  }
  std::vector<float> a(T * in);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t j = 0; j < in; ++j) {
      a[t * in + j] = (j < 32 ? 1.f : 10.f) * static_cast<float>(t + 1);
    }
  }
  std::vector<float> out(T * out_dim, 99.f);

  engine.MatMulT(AsMutableMat(out, T, out_dim), AsMat(a, T, in),
                 AsQ8Mat(blocks, out_dim, in));

  for (int64_t t = 0; t < T; ++t) {
    EXPECT_FLOAT_EQ(out[t * out_dim + 0], 672.f * (t + 1)) << "t=" << t;
    EXPECT_FLOAT_EQ(out[t * out_dim + 1], 64.f * (t + 1)) << "t=" << t;
  }
}

// Q8 and F32 MatMulT on the same logical weights must agree for a multi-row
// input. Scale = 1 so the only difference is accumulation order.
TEST(ComputeEngineMatMulTTest, Q8MatchesF32MultiRow) {
  ComputeEngine engine;
  const int64_t T = 2, rows = 3, cols = 64;
  std::vector<BlockQ8_0> blocks(rows * 2);
  std::vector<float> w_f32(rows * cols);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t b = 0; b < 2; ++b) {
      BlockQ8_0& block = blocks[r * 2 + b];
      block.scale = 1.f;
      for (int64_t i = 0; i < 32; ++i) {
        const int8_t q = static_cast<int8_t>((r * 7 + b * 11 + i * 3) % 21 - 10);
        block.data[i] = q;
        w_f32[r * cols + b * 32 + i] = static_cast<float>(q);
      }
    }
  }
  std::vector<float> a(T * cols);
  for (int64_t i = 0; i < T * cols; ++i) {
    a[i] = 0.25f * static_cast<float>((i * 5) % 17 - 8);
  }

  std::vector<float> out_q8(T * rows, 99.f), out_f32(T * rows, -99.f);
  engine.MatMulT(AsMutableMat(out_q8, T, rows), AsMat(a, T, cols),
                 AsQ8Mat(blocks, rows, cols));
  engine.MatMulT(AsMutableMat(out_f32, T, rows), AsMat(a, T, cols),
                 AsMat(w_f32, rows, cols));

  for (int64_t i = 0; i < T * rows; ++i) {
    EXPECT_NEAR(out_q8[i], out_f32[i], 1e-3) << "i=" << i;
  }
}

// ── Copy ──────────────────────────────────────────────────────────────────────
// F32 src: plain copy. Q8_0 src: dequantize into the F32 dst.

TEST(ComputeEngineCopyTest, F32CopiesValues) {
  ComputeEngine engine;
  const std::vector<float> src_orig = {1.f, -2.5f, 0.f, 42.f};
  std::vector<float> src = src_orig;
  std::vector<float> dst(4, 99.f);  // sentinel: must be overwritten

  engine.Copy(AsMutableVec(dst), AsVec(src));

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(dst[i], src_orig[i]) << "i=" << i;
    EXPECT_FLOAT_EQ(src[i], src_orig[i]) << "src modified at i=" << i;
  }
}

// Q8_0 source: out[b*32 + i] = scale_b * data_b[i], including negatives.
TEST(ComputeEngineCopyTest, Q8DequantHandComputed) {
  ComputeEngine engine;
  std::vector<BlockQ8_0> blocks(2);
  blocks[0].scale = 0.5f;
  blocks[1].scale = 2.f;
  for (int i = 0; i < 32; ++i) {
    blocks[0].data[i] = static_cast<int8_t>(i - 16);  // negatives included
    blocks[1].data[i] = -3;
  }
  std::vector<float> dst(64, 99.f);

  VectorView q8 = VectorView::Create(
      DType::Q8_0, reinterpret_cast<const std::byte*>(blocks.data()), 64);
  engine.Copy(AsMutableVec(dst), q8);

  EXPECT_FLOAT_EQ(dst[0], -8.f);    // 0.5 * (0 - 16)
  EXPECT_FLOAT_EQ(dst[16], 0.f);    // 0.5 * 0
  EXPECT_FLOAT_EQ(dst[31], 7.5f);   // 0.5 * 15
  for (int i = 32; i < 64; ++i) {
    EXPECT_FLOAT_EQ(dst[i], -6.f) << "i=" << i;  // 2 * -3
  }
}

// Cross-check against the Q8 matmul: dequantizing a row and dotting it with x
// must match MatMul of the same row (up to accumulation-order rounding).
TEST(ComputeEngineCopyTest, Q8DequantMatchesMatMul) {
  ComputeEngine engine;
  std::vector<BlockQ8_0> blocks(2);  // one row, in = 64
  blocks[0].scale = 0.25f;
  blocks[1].scale = 1.5f;
  for (int i = 0; i < 32; ++i) {
    blocks[0].data[i] = static_cast<int8_t>((i * 5) % 17 - 8);
    blocks[1].data[i] = static_cast<int8_t>((i * 3) % 11 - 5);
  }
  std::vector<float> x(64);
  for (int i = 0; i < 64; ++i) x[i] = 0.1f * static_cast<float>(i % 7 - 3);

  // Path 1: MatMul on the 1x64 Q8 matrix.
  std::vector<float> matmul_out = {99.f};
  engine.MatMul(AsMutableVec(matmul_out), AsQ8Mat(blocks, 1, 64), AsVec(x));

  // Path 2: Copy-dequant the row, then dot manually.
  std::vector<float> dequant(64, 99.f);
  engine.Copy(AsMutableVec(dequant), AsQ8Mat(blocks, 1, 64).At(0));
  float dot = 0.f;
  for (int i = 0; i < 64; ++i) dot += dequant[i] * x[i];

  EXPECT_NEAR(matmul_out[0], dot, 1e-4);
}

// ── RmsNorm ───────────────────────────────────────────────────────────────────
// Row-wise: each row of the [T, d] input is normalized by its own RMS and
// scaled by the shared gamma.

constexpr float kRmsEps = 1e-5f;

// x = {3, 4}: mean of squares = 12.5, rms = sqrt(12.5) ≈ 3.5355339.
// With gamma = {1, 2}: out = {3/rms, 2*4/rms}.
TEST(ComputeEngineRmsNormTest, HandComputed) {
  ComputeEngine engine;
  std::vector<float> x = {3.f, 4.f};
  std::vector<float> gamma = {1.f, 2.f};
  std::vector<float> out = {99.f, 99.f};  // sentinel: must overwrite

  engine.RmsNorm(AsMutableRow(out), AsRow(x), AsVec(gamma), /*epsilon=*/0.f);

  EXPECT_NEAR(out[0], 0.84852814f, 1e-6);
  EXPECT_NEAR(out[1], 2.26274170f, 1e-6);
}

// rms(c*x) = c*rms(x), so the c cancels: scaling the input must not change
// the output (up to epsilon). The defining property of the normalization.
TEST(ComputeEngineRmsNormTest, ScaleInvariant) {
  ComputeEngine engine;
  const std::vector<float> x = {0.5f, -1.5f, 2.f, 0.25f};
  const std::vector<float> gamma = {1.f, 0.5f, 2.f, -1.f};

  std::vector<float> base(4), scaled_in(4), scaled_out(4);
  engine.RmsNorm(AsMutableRow(base), AsRow(x), AsVec(gamma), kRmsEps);

  for (float c : {2.f, 100.f}) {
    for (int i = 0; i < 4; ++i) scaled_in[i] = c * x[i];
    engine.RmsNorm(AsMutableRow(scaled_out), AsRow(scaled_in), AsVec(gamma),
                   kRmsEps);
    for (int i = 0; i < 4; ++i) {
      EXPECT_NEAR(scaled_out[i], base[i], 1e-4) << "c=" << c << " i=" << i;
    }
  }
}

// With gamma = 1 the output must have RMS ≈ 1 — normalization actually
// normalizes.
TEST(ComputeEngineRmsNormTest, GammaOneGivesUnitRms) {
  ComputeEngine engine;
  std::vector<float> x = {10.f, -20.f, 5.f, 0.f, 7.5f, -1.f, 3.f, 40.f};
  std::vector<float> gamma(8, 1.f);
  std::vector<float> out(8, 0.f);

  engine.RmsNorm(AsMutableRow(out), AsRow(x), AsVec(gamma), kRmsEps);

  float mean_sq = 0.f;
  for (float v : out) mean_sq += v * v;
  mean_sq /= out.size();
  EXPECT_NEAR(mean_sq, 1.f, 1e-4);
}

// RmsNorm must not modify its input — DecoderBlock reuses the un-normed input
// for the residual add.
TEST(ComputeEngineRmsNormTest, InputUntouched) {
  ComputeEngine engine;
  const std::vector<float> orig = {1.f, -2.f, 3.f, -4.f};
  std::vector<float> x = orig;
  std::vector<float> gamma = {1.f, 1.f, 1.f, 1.f};
  std::vector<float> out(4);

  engine.RmsNorm(AsMutableRow(out), AsRow(x), AsVec(gamma), kRmsEps);

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(x[i], orig[i]) << "i=" << i;
  }
}

// All-zero input: epsilon must prevent 0/0 — output is zeros, not NaN.
TEST(ComputeEngineRmsNormTest, ZeroInputNoNan) {
  ComputeEngine engine;
  std::vector<float> x(4, 0.f);
  std::vector<float> gamma(4, 1.f);
  std::vector<float> out(4, 99.f);

  engine.RmsNorm(AsMutableRow(out), AsRow(x), AsVec(gamma), kRmsEps);

  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(std::isfinite(out[i])) << "i=" << i;
    EXPECT_FLOAT_EQ(out[i], 0.f) << "i=" << i;
  }
}

// Two rows with different content: each is normalized by ITS OWN rms (both
// rows here have rms = sqrt(12.5)) and multiplied by the same gamma. Catches
// a loop that normalizes row 0 twice or pools statistics across rows.
TEST(ComputeEngineRmsNormTest, RowsNormalizedIndependently) {
  ComputeEngine engine;
  std::vector<float> x = {3.f, 4.f,
                          4.f, 3.f};
  std::vector<float> gamma = {1.f, 2.f};
  std::vector<float> out(4, 99.f);

  engine.RmsNorm(AsMutableMat(out, 2, 2), AsMat(x, 2, 2), AsVec(gamma),
                 /*epsilon=*/0.f);

  EXPECT_NEAR(out[0], 0.84852814f, 1e-6);  // 3 / rms
  EXPECT_NEAR(out[1], 2.26274170f, 1e-6);  // 2 * 4 / rms
  EXPECT_NEAR(out[2], 1.13137085f, 1e-6);  // 4 / rms
  EXPECT_NEAR(out[3], 1.69705627f, 1e-6);  // 2 * 3 / rms
}

// Batched rows must equal the same rows normalized one at a time — the
// prefill/decode equivalence for this kernel. Rows have different norms so a
// shared statistic would show.
TEST(ComputeEngineRmsNormTest, BatchedMatchesSingleRow) {
  ComputeEngine engine;
  const int64_t T = 3, d = 4;
  std::vector<float> x = {
      0.5f, -1.5f, 2.f,   0.25f,
      10.f, 20.f,  -5.f,  1.f,
      -0.1f, 0.2f, 0.3f,  -0.4f,
  };
  std::vector<float> gamma = {1.f, 0.5f, 2.f, -1.f};

  std::vector<float> batched(T * d, 99.f);
  engine.RmsNorm(AsMutableMat(batched, T, d), AsMat(x, T, d), AsVec(gamma),
                 kRmsEps);

  for (int64_t t = 0; t < T; ++t) {
    std::vector<float> single(d, -99.f);
    engine.RmsNorm(AsMutableRow(single), AsMat(x, T, d).Slice(t, 1),
                   AsVec(gamma), kRmsEps);
    for (int64_t i = 0; i < d; ++i) {
      EXPECT_FLOAT_EQ(batched[t * d + i], single[i]) << "t=" << t << " i=" << i;
    }
  }
}

// ── SwiGluMul ─────────────────────────────────────────────────────────────────
// gate = SiLU(gate) ⊙ up, in place on gate. SiLU(z) = z / (1 + e^-z).

// up = 1 reduces the op to pure SiLU: pins the activation values themselves.
TEST(ComputeEngineSwiGluMulTest, PureSiluWithUnitUp) {
  ComputeEngine engine;
  std::vector<float> gate = {0.f, 1.f, -1.f, 2.f};
  std::vector<float> up(4, 1.f);

  engine.SwiGluMul(AsMutableRow(gate), AsRow(up));

  EXPECT_NEAR(gate[0], 0.f, 1e-6);
  EXPECT_NEAR(gate[1], 0.7310586f, 1e-6);   // 1 * sigmoid(1)
  EXPECT_NEAR(gate[2], -0.2689414f, 1e-6);  // -1 * sigmoid(-1)
  EXPECT_NEAR(gate[3], 1.7615942f, 1e-6);   // 2 * sigmoid(2)
}

// SiLU must apply to the GATE operand, not up. gate={1,2}, up={2,1} gives
// {silu(1)*2, silu(2)*1}; a swapped implementation gives the reversed pair.
TEST(ComputeEngineSwiGluMulTest, SiluAppliesToGateNotUp) {
  ComputeEngine engine;
  std::vector<float> gate = {1.f, 2.f};
  std::vector<float> up = {2.f, 1.f};

  engine.SwiGluMul(AsMutableRow(gate), AsRow(up));

  EXPECT_NEAR(gate[0], 1.4621172f, 1e-6);   // silu(1) * 2
  EXPECT_NEAR(gate[1], 1.7615942f, 1e-6);   // silu(2) * 1
}

// A zero gate closes the channel completely, regardless of up's magnitude.
TEST(ComputeEngineSwiGluMulTest, ZeroGateClosesChannel) {
  ComputeEngine engine;
  std::vector<float> gate = {0.f, 0.f};
  std::vector<float> up = {1e6f, -1e6f};

  engine.SwiGluMul(AsMutableRow(gate), AsRow(up));

  EXPECT_FLOAT_EQ(gate[0], 0.f);
  EXPECT_FLOAT_EQ(gate[1], 0.f);
}

// The up operand is read-only; only gate is modified.
TEST(ComputeEngineSwiGluMulTest, UpUntouched) {
  ComputeEngine engine;
  std::vector<float> gate = {1.f, -2.f, 3.f};
  const std::vector<float> up_orig = {0.5f, 2.f, -1.f};
  std::vector<float> up = up_orig;

  engine.SwiGluMul(AsMutableRow(gate), AsRow(up));

  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(up[i], up_orig[i]) << "i=" << i;
  }
}

// Large magnitudes must stay finite: exp overflow/underflow lands on the
// correct SiLU limits (0 for very negative, identity for very positive).
TEST(ComputeEngineSwiGluMulTest, ExtremeInputsFinite) {
  ComputeEngine engine;
  std::vector<float> gate = {-100.f, 100.f};
  std::vector<float> up = {1.f, 1.f};

  engine.SwiGluMul(AsMutableRow(gate), AsRow(up));

  EXPECT_TRUE(std::isfinite(gate[0]));
  EXPECT_TRUE(std::isfinite(gate[1]));
  EXPECT_NEAR(gate[0], 0.f, 1e-5);      // silu(-100) ≈ 0
  EXPECT_NEAR(gate[1], 100.f, 1e-3);    // silu(100) ≈ 100
}

// Multi-row: purely elementwise, so every (row, col) pairs gate with its own
// up entry. Row 1 reuses the values from the single-row tests above.
TEST(ComputeEngineSwiGluMulTest, MultiRowElementwise) {
  ComputeEngine engine;
  std::vector<float> gate = {1.f, 2.f,
                             0.f, -1.f};
  std::vector<float> up = {2.f, 1.f,
                           5.f, 1.f};

  engine.SwiGluMul(AsMutableMat(gate, 2, 2), AsMat(up, 2, 2));

  EXPECT_NEAR(gate[0], 1.4621172f, 1e-6);   // silu(1) * 2
  EXPECT_NEAR(gate[1], 1.7615942f, 1e-6);   // silu(2) * 1
  EXPECT_NEAR(gate[2], 0.f, 1e-6);          // silu(0) * 5
  EXPECT_NEAR(gate[3], -0.2689414f, 1e-6);  // silu(-1) * 1
}

// ── Softmax ───────────────────────────────────────────────────────────────────

// Probabilities sum to 1, preserve ordering, and e^1 / e^0 = e between
// adjacent logits.
TEST(ComputeEngineSoftmaxTest, HandComputed) {
  ComputeEngine engine;
  std::vector<float> v = {0.f, 1.f, 0.f};

  engine.Softmax(AsMutableVec(v));

  const float e = std::exp(1.f);
  EXPECT_NEAR(v[0], 1.f / (2.f + e), 1e-6);
  EXPECT_NEAR(v[1], e / (2.f + e), 1e-6);
  EXPECT_NEAR(v[2], 1.f / (2.f + e), 1e-6);
  EXPECT_NEAR(v[0] + v[1] + v[2], 1.f, 1e-6);
}

// Shift invariance and overflow safety: adding a constant (even a huge one)
// to every logit must not change the result — the max-subtraction trick.
TEST(ComputeEngineSoftmaxTest, ShiftInvariantAndFinite) {
  ComputeEngine engine;
  std::vector<float> base = {1.f, 2.f, -1.f};
  std::vector<float> shifted = {1001.f, 1002.f, 999.f};

  engine.Softmax(AsMutableVec(base));
  engine.Softmax(AsMutableVec(shifted));

  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(std::isfinite(shifted[i])) << "i=" << i;
    EXPECT_NEAR(shifted[i], base[i], 1e-6) << "i=" << i;
  }
}

// ── Rope ──────────────────────────────────────────────────────────────────────
// Row-wise with a per-row position: row r of the [T, d] input is rotated at
// position `position + r`.

constexpr float kFreqBase = 10000.f;

// Position 0 means angle 0 for every pair: the vector must be unchanged.
TEST(ComputeEngineRopeTest, PositionZeroIsIdentity) {
  ComputeEngine engine;
  std::vector<float> v = {1.f, 2.f, -3.f, 0.5f, 0.f, -1.f, 7.f, 0.25f};
  const std::vector<float> orig = v;
  const std::vector<float> unit_freqs = {1.f, 1.f};

  engine.Rope(AsMutableRow(v), /*position=*/0, kFreqBase,
              /*dimension_count=*/4, AsVec(unit_freqs));

  for (size_t i = 0; i < v.size(); ++i) {
    EXPECT_FLOAT_EQ(v[i], orig[i]) << "i=" << i;
  }
}

// With head_dim = 2 there is a single pair whose theta = base^0 = 1, so the
// angle is exactly `position` radians. Rotating (1, 0) by m radians must give
// (cos m, sin m) — this pins the interleaved pairing, the rotation direction,
// and the sin/cos placement.
TEST(ComputeEngineRopeTest, HandComputedSinglePair) {
  ComputeEngine engine;
  const int64_t m = 2;
  std::vector<float> v = {1.f, 0.f};
  const std::vector<float> unit_freqs = {1.f};

  engine.Rope(AsMutableRow(v), m, kFreqBase, /*dimension_count=*/2,
              AsVec(unit_freqs));

  EXPECT_NEAR(v[0], std::cos(2.0), 1e-6);
  EXPECT_NEAR(v[1], std::sin(2.0), 1e-6);
}

// A rotation never changes a vector's length, at any position.
TEST(ComputeEngineRopeTest, PreservesNorm) {
  ComputeEngine engine;
  const std::vector<float> orig = {0.3f, -1.7f, 2.2f, 0.9f,
                                   -0.4f, 5.f,  -2.f, 1.1f};
  float norm2 = 0.f;
  for (float x : orig) norm2 += x * x;

  const std::vector<float> unit_freqs = {1.f, 1.f};
  for (int64_t pos : {1, 17, 4096}) {
    std::vector<float> v = orig;
    engine.Rope(AsMutableRow(v), pos, kFreqBase, /*dimension_count=*/4,
                AsVec(unit_freqs));

    float got = 0.f;
    for (float x : v) got += x * x;
    EXPECT_NEAR(got, norm2, 1e-3) << "pos=" << pos;
  }
}

// The defining RoPE property: the dot product of a roped q at position m and a
// roped k at position n depends only on n - m. Shifting both positions by the
// same delta must not change the score.
TEST(ComputeEngineRopeTest, DotDependsOnRelativePositionOnly) {
  ComputeEngine engine;
  const std::vector<float> q0 = {0.7f, -0.3f, 1.2f, 0.4f};
  const std::vector<float> k0 = {-1.1f, 0.6f, 0.2f, 2.f};
  const int64_t dim = 4;
  // Non-uniform factors: the relative-position property must hold for ANY
  // frequency schedule, scaled or not.
  const std::vector<float> freqs = {1.f, 2.f};

  auto roped_dot = [&](int64_t pos_q, int64_t pos_k) {
    std::vector<float> q = q0;
    std::vector<float> k = k0;
    engine.Rope(AsMutableRow(q), pos_q, kFreqBase, dim, AsVec(freqs));
    engine.Rope(AsMutableRow(k), pos_k, kFreqBase, dim, AsVec(freqs));
    float dot = 0.f;
    for (size_t i = 0; i < q.size(); ++i) dot += q[i] * k[i];
    return dot;
  };

  const float base_dot = roped_dot(3, 7);         // distance 4
  EXPECT_NEAR(roped_dot(0, 4), base_dot, 1e-4);   // same distance, shifted
  EXPECT_NEAR(roped_dot(100, 104), base_dot, 1e-4);
  // Different distance must (generically) give a different score.
  EXPECT_GT(std::abs(roped_dot(3, 8) - base_dot), 1e-3);
}

// The frequency schedule restarts every head: two heads with identical content
// must transform identically. Catches using the global pair index for theta.
TEST(ComputeEngineRopeTest, PatternRepeatsPerHead) {
  ComputeEngine engine;
  const std::vector<float> head = {0.5f, -1.f, 2.f, 0.25f};
  std::vector<float> v;
  v.insert(v.end(), head.begin(), head.end());
  v.insert(v.end(), head.begin(), head.end());  // two identical heads

  const std::vector<float> unit_freqs = {1.f, 1.f};
  engine.Rope(AsMutableRow(v), /*position=*/9, kFreqBase,
              /*dimension_count=*/4, AsVec(unit_freqs));

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(v[i], v[4 + i]) << "i=" << i;
  }
}

// rope_freqs divides each pair's theta. With head_dim 4 and freq_base 4:
// theta_0 = 1, theta_1 = 4^(-1/2) = 0.5. Factors {1, 2} leave pair 0's angle
// at position*1 and halve pair 1's to position*0.25.
TEST(ComputeEngineRopeTest, FreqFactorsScaleAngles) {
  ComputeEngine engine;
  std::vector<float> v = {1.f, 0.f, 1.f, 0.f};
  const std::vector<float> freqs = {1.f, 2.f};

  engine.Rope(AsMutableRow(v), /*position=*/2, /*freq_base=*/4.f,
              /*dimension_count=*/4, AsVec(freqs));

  EXPECT_NEAR(v[0], std::cos(2.0), 1e-6);   // pair 0: angle = 2 * 1 / 1
  EXPECT_NEAR(v[1], std::sin(2.0), 1e-6);
  EXPECT_NEAR(v[2], std::cos(0.5), 1e-6);   // pair 1: angle = 2 * 0.5 / 2
  EXPECT_NEAR(v[3], std::sin(0.5), 1e-6);
}

// Prefill: row r must be rotated at position `start + r`. Three identical rows
// at start = 5 must equal single-row Rope at positions 5, 6, 7 — and differ
// from each other, proving the offset is applied rather than a shared
// position.
TEST(ComputeEngineRopeTest, RowsUseStartPositionPlusRowIndex) {
  ComputeEngine engine;
  const int64_t T = 3, start = 5;
  const std::vector<float> row = {0.5f, -1.f, 2.f, 0.25f};
  std::vector<float> batched;
  for (int64_t t = 0; t < T; ++t) {
    batched.insert(batched.end(), row.begin(), row.end());
  }
  const std::vector<float> freqs = {1.f, 2.f};

  engine.Rope(AsMutableMat(batched, T, 4), start, kFreqBase,
              /*dimension_count=*/4, AsVec(freqs));

  for (int64_t t = 0; t < T; ++t) {
    std::vector<float> single = row;
    engine.Rope(AsMutableRow(single), start + t, kFreqBase,
                /*dimension_count=*/4, AsVec(freqs));
    for (int i = 0; i < 4; ++i) {
      EXPECT_FLOAT_EQ(batched[t * 4 + i], single[i]) << "t=" << t << " i=" << i;
    }
  }
  // Rows at different positions must actually differ.
  EXPECT_GT(std::abs(batched[0] - batched[4]), 1e-3);
  EXPECT_GT(std::abs(batched[4] - batched[8]), 1e-3);
}

// A single row in a [1, d] matrix at position p equals a matrix whose row r
// is at position p - r... i.e. the position argument is the position of ROW 0,
// not of the last row. Start = 0 makes row 0 the identity and row 1 rotated.
TEST(ComputeEngineRopeTest, PositionArgumentIsRowZero) {
  ComputeEngine engine;
  std::vector<float> v = {1.f, 0.f,
                          1.f, 0.f};
  const std::vector<float> unit_freqs = {1.f};

  engine.Rope(AsMutableMat(v, 2, 2), /*position=*/0, kFreqBase,
              /*dimension_count=*/2, AsVec(unit_freqs));

  EXPECT_FLOAT_EQ(v[0], 1.f);                 // row 0: position 0, identity
  EXPECT_FLOAT_EQ(v[1], 0.f);
  EXPECT_NEAR(v[2], std::cos(1.0), 1e-6);     // row 1: position 1
  EXPECT_NEAR(v[3], std::sin(1.0), 1e-6);
}

}  // namespace
}  // namespace tlm
