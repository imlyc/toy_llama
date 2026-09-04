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

// ── MatMul ────────────────────────────────────────────────────────────────────
// Convention: out = mat @ vec with mat shaped [out, in], rows contiguous
// (GGUF weight layout; Q8_0 blocks run along the summed `in` dimension).

MatrixView AsQ8Mat(const std::vector<BlockQ8_0>& blocks, int64_t rows,
                   int64_t cols) {
  return MatrixView::Create(DType::Q8_0,
                            reinterpret_cast<const std::byte*>(blocks.data()),
                            rows, cols);
}

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

// ── RmsNorm ───────────────────────────────────────────────────────────────────

constexpr float kRmsEps = 1e-5f;

// x = {3, 4}: mean of squares = 12.5, rms = sqrt(12.5) ≈ 3.5355339.
// With gamma = {1, 2}: out = {3/rms, 2*4/rms}.
TEST(ComputeEngineRmsNormTest, HandComputed) {
  ComputeEngine engine;
  std::vector<float> x = {3.f, 4.f};
  std::vector<float> gamma = {1.f, 2.f};
  std::vector<float> out = {99.f, 99.f};  // sentinel: must overwrite

  engine.RmsNorm(AsMutableVec(out), AsVec(x), AsVec(gamma), /*epsilon=*/0.f);

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
  engine.RmsNorm(AsMutableVec(base), AsVec(x), AsVec(gamma), kRmsEps);

  for (float c : {2.f, 100.f}) {
    for (int i = 0; i < 4; ++i) scaled_in[i] = c * x[i];
    engine.RmsNorm(AsMutableVec(scaled_out), AsVec(scaled_in), AsVec(gamma),
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

  engine.RmsNorm(AsMutableVec(out), AsVec(x), AsVec(gamma), kRmsEps);

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

  engine.RmsNorm(AsMutableVec(out), AsVec(x), AsVec(gamma), kRmsEps);

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

  engine.RmsNorm(AsMutableVec(out), AsVec(x), AsVec(gamma), kRmsEps);

  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(std::isfinite(out[i])) << "i=" << i;
    EXPECT_FLOAT_EQ(out[i], 0.f) << "i=" << i;
  }
}

// ── SwiGluMul ─────────────────────────────────────────────────────────────────
// gate = SiLU(gate) ⊙ up, in place on gate. SiLU(z) = z / (1 + e^-z).

// up = 1 reduces the op to pure SiLU: pins the activation values themselves.
TEST(ComputeEngineSwiGluMulTest, PureSiluWithUnitUp) {
  ComputeEngine engine;
  std::vector<float> gate = {0.f, 1.f, -1.f, 2.f};
  std::vector<float> up(4, 1.f);

  engine.SwiGluMul(AsMutableVec(gate), AsVec(up));

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

  engine.SwiGluMul(AsMutableVec(gate), AsVec(up));

  EXPECT_NEAR(gate[0], 1.4621172f, 1e-6);   // silu(1) * 2
  EXPECT_NEAR(gate[1], 1.7615942f, 1e-6);   // silu(2) * 1
}

// A zero gate closes the channel completely, regardless of up's magnitude.
TEST(ComputeEngineSwiGluMulTest, ZeroGateClosesChannel) {
  ComputeEngine engine;
  std::vector<float> gate = {0.f, 0.f};
  std::vector<float> up = {1e6f, -1e6f};

  engine.SwiGluMul(AsMutableVec(gate), AsVec(up));

  EXPECT_FLOAT_EQ(gate[0], 0.f);
  EXPECT_FLOAT_EQ(gate[1], 0.f);
}

// The up operand is read-only; only gate is modified.
TEST(ComputeEngineSwiGluMulTest, UpUntouched) {
  ComputeEngine engine;
  std::vector<float> gate = {1.f, -2.f, 3.f};
  const std::vector<float> up_orig = {0.5f, 2.f, -1.f};
  std::vector<float> up = up_orig;

  engine.SwiGluMul(AsMutableVec(gate), AsVec(up));

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

  engine.SwiGluMul(AsMutableVec(gate), AsVec(up));

  EXPECT_TRUE(std::isfinite(gate[0]));
  EXPECT_TRUE(std::isfinite(gate[1]));
  EXPECT_NEAR(gate[0], 0.f, 1e-5);      // silu(-100) ≈ 0
  EXPECT_NEAR(gate[1], 100.f, 1e-3);    // silu(100) ≈ 100
}

// ── Rope ──────────────────────────────────────────────────────────────────────

constexpr float kFreqBase = 10000.f;

// Position 0 means angle 0 for every pair: the vector must be unchanged.
TEST(ComputeEngineRopeTest, PositionZeroIsIdentity) {
  ComputeEngine engine;
  std::vector<float> v = {1.f, 2.f, -3.f, 0.5f, 0.f, -1.f, 7.f, 0.25f};
  const std::vector<float> orig = v;
  const std::vector<float> unit_freqs = {1.f, 1.f};

  engine.Rope(AsMutableVec(v), /*position=*/0, kFreqBase,
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

  engine.Rope(AsMutableVec(v), m, kFreqBase, /*dimension_count=*/2,
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
    engine.Rope(AsMutableVec(v), pos, kFreqBase, /*dimension_count=*/4,
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
    engine.Rope(AsMutableVec(q), pos_q, kFreqBase, dim, AsVec(freqs));
    engine.Rope(AsMutableVec(k), pos_k, kFreqBase, dim, AsVec(freqs));
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
  engine.Rope(AsMutableVec(v), /*position=*/9, kFreqBase,
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

  engine.Rope(AsMutableVec(v), /*position=*/2, /*freq_base=*/4.f,
              /*dimension_count=*/4, AsVec(freqs));

  EXPECT_NEAR(v[0], std::cos(2.0), 1e-6);   // pair 0: angle = 2 * 1 / 1
  EXPECT_NEAR(v[1], std::sin(2.0), 1e-6);
  EXPECT_NEAR(v[2], std::cos(0.5), 1e-6);   // pair 1: angle = 2 * 0.5 / 2
  EXPECT_NEAR(v[3], std::sin(0.5), 1e-6);
}

}  // namespace
}  // namespace tlm
