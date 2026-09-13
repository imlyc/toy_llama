#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "compute/compute_engine.h"
#include "tensor/dtype.h"
#include "tensor/tensor_view.h"
#include "transformer/decoder_block.h"
#include "transformer/group_query_attn_block.h"
#include "transformer/rms_norm_layer.h"
#include "transformer/swiglu_ffn_block.h"

namespace tlm {
namespace {

VectorView AsVec(const std::vector<float>& v) {
  return VectorView::Create(DType::F32,
                            reinterpret_cast<const std::byte*>(v.data()),
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

// Single-row [1, n] views: the decode (T = 1) case of every layer.
MatrixView AsRow(const std::vector<float>& v) {
  return AsMat(v, 1, static_cast<int64_t>(v.size()));
}

// Layer outputs live in layer-owned buffers that the next Forward overwrites,
// so results that must survive another call are copied out.
std::vector<float> ToVector(MatrixView m) {
  std::span<const float> s = m.As<const float>();
  return std::vector<float>(s.begin(), s.end());
}

constexpr float kEps = 1e-5f;

const std::vector<float> kIdentity2 = {1.f, 0.f, 0.f, 1.f};
const std::vector<float> kSwap2 = {0.f, 1.f, 1.f, 0.f};
const std::vector<float> kZero2 = {0.f, 0.f, 0.f, 0.f};
const std::vector<float> kOnes2 = {1.f, 1.f};
const std::vector<float> kUnitRopeFreqs = {1.f};

// ── RmsNormLayer ──────────────────────────────────────────────────────────────

// x = {3, 4}: rms = sqrt(12.5 + eps); gamma = {1, 2}.
TEST(RmsNormLayerTest, ForwardHandComputed) {
  ComputeEngine engine;
  const std::vector<float> gamma = {1.f, 2.f};
  RmsNormLayer layer(engine, {AsVec(gamma), 0.f});

  const std::vector<float> x = {3.f, 4.f};
  MatrixView out = layer.Forward(AsRow(x));

  ASSERT_EQ(out.shape[0], 1);
  ASSERT_EQ(out.shape[1], 2);
  std::span<const float> data = out.As<const float>();
  EXPECT_NEAR(data[0], 0.84852814f, 1e-6);
  EXPECT_NEAR(data[1], 2.26274170f, 1e-6);
}

// The layer owns its output buffer: repeated Forward calls overwrite it and
// give identical results for identical input.
TEST(RmsNormLayerTest, RepeatedForwardIsIdempotent) {
  ComputeEngine engine;
  const std::vector<float> gamma = {1.f, 1.f};
  RmsNormLayer layer(engine, {AsVec(gamma), kEps});

  const std::vector<float> x = {1.f, -2.f};
  const std::vector<float> first = ToVector(layer.Forward(AsRow(x)));
  const std::vector<float> second = ToVector(layer.Forward(AsRow(x)));

  EXPECT_FLOAT_EQ(second[0], first[0]);
  EXPECT_FLOAT_EQ(second[1], first[1]);
}

// A batched [T, d] forward must equal T single-row forwards, and the output
// shape must follow the input's row count (buffer grows/shrinks with T).
TEST(RmsNormLayerTest, BatchedMatchesSingleRow) {
  ComputeEngine engine;
  const std::vector<float> gamma = {1.f, 2.f};
  RmsNormLayer layer(engine, {AsVec(gamma), kEps});

  const std::vector<float> x = {3.f, 4.f,
                                -1.f, 0.5f,
                                10.f, 10.f};
  MatrixView batched_view = layer.Forward(AsMat(x, 3, 2));
  ASSERT_EQ(batched_view.shape[0], 3);
  ASSERT_EQ(batched_view.shape[1], 2);
  const std::vector<float> batched = ToVector(batched_view);

  for (int64_t t = 0; t < 3; ++t) {
    const std::vector<float> single =
        ToVector(layer.Forward(AsMat(x, 3, 2).Slice(t, 1)));
    EXPECT_FLOAT_EQ(batched[t * 2], single[0]) << "t=" << t;
    EXPECT_FLOAT_EQ(batched[t * 2 + 1], single[1]) << "t=" << t;
  }
}

// ── SwiGluFfnBlock ────────────────────────────────────────────────────────────

// x = {1, 2}; wup = I → up = {1, 2}; wgate = swap → gate = {2, 1};
// SwiGluMul: {silu(2)*1, silu(1)*2}; wdown = [[1,2],[3,4]] mixes the result.
// Catches gate/up swaps and weight orientation mistakes.
TEST(SwiGluFfnBlockTest, ForwardHandComputed) {
  ComputeEngine engine;
  const std::vector<float> wdown = {1.f, 2.f, 3.f, 4.f};
  SwiGluFfnBlock block(engine, {.wup = AsMat(kIdentity2, 2, 2),
                                .wgate = AsMat(kSwap2, 2, 2),
                                .wdown = AsMat(wdown, 2, 2)});

  const std::vector<float> x = {1.f, 2.f};
  std::span<const float> out = block.Forward(AsRow(x)).As<const float>();

  // gate after SwiGluMul: {1.7615942, 1.4621172}
  EXPECT_NEAR(out[0], 4.6858287f, 1e-4);   // 1.7615942 + 2 * 1.4621172
  EXPECT_NEAR(out[1], 11.133251f, 1e-4);   // 3 * 1.7615942 + 4 * 1.4621172
}

// The FFN is stateless and row-wise: a [T, d] forward equals T single-row
// forwards. Rows are chosen so gate/up differ per row.
TEST(SwiGluFfnBlockTest, BatchedMatchesSingleRow) {
  ComputeEngine engine;
  const std::vector<float> wdown = {1.f, 2.f, 3.f, 4.f};
  SwiGluFfnBlock block(engine, {.wup = AsMat(kIdentity2, 2, 2),
                                .wgate = AsMat(kSwap2, 2, 2),
                                .wdown = AsMat(wdown, 2, 2)});

  const std::vector<float> x = {1.f, 2.f,
                                -1.f, 0.5f,
                                3.f, -2.f};
  const std::vector<float> batched = ToVector(block.Forward(AsMat(x, 3, 2)));

  for (int64_t t = 0; t < 3; ++t) {
    const std::vector<float> single =
        ToVector(block.Forward(AsMat(x, 3, 2).Slice(t, 1)));
    EXPECT_NEAR(batched[t * 2], single[0], 1e-5) << "t=" << t;
    EXPECT_NEAR(batched[t * 2 + 1], single[1], 1e-5) << "t=" << t;
  }
}

// ── GroupQueryAttnBlock ───────────────────────────────────────────────────────

GroupQueryAttnBlock::Param TinyGqaParam(MatrixView wq, MatrixView wk,
                                        MatrixView wv, MatrixView wo) {
  GroupQueryAttnBlock::Param param;
  param.key_size = 2;
  param.value_size = 2;
  param.context_length = 4;
  param.head_count = 1;
  param.head_count_kv = 1;
  param.rope_freq_base = 10000.f;
  param.rope_dimension_count = 2;
  param.wq = wq;
  param.wk = wk;
  param.wv = wv;
  param.wo = wo;
  param.rope_freqs = AsVec(kUnitRopeFreqs);
  return param;
}

// Non-degenerate weights for the batched-vs-serial checks: q and k are
// non-zero (so RoPE positions and the causal mask matter) and wo mixes.
GroupQueryAttnBlock::Param MixingGqaParam(const std::vector<float>& wo) {
  return TinyGqaParam(AsMat(kIdentity2, 2, 2), AsMat(kSwap2, 2, 2),
                      AsMat(kIdentity2, 2, 2), AsMat(wo, 2, 2));
}

// First token: position 0 makes RoPE the identity, and softmax over a single
// cached key is 1 regardless of q/k. So output = wo @ (wv @ x).
TEST(GroupQueryAttnBlockTest, SingleTokenPassesValueThroughWo) {
  ComputeEngine engine;
  GroupQueryAttnBlock block(
      engine, TinyGqaParam(AsMat(kIdentity2, 2, 2), AsMat(kIdentity2, 2, 2),
                           AsMat(kIdentity2, 2, 2), AsMat(kSwap2, 2, 2)));

  const std::vector<float> x = {1.f, 2.f};
  std::span<const float> out = block.Forward(AsRow(x)).As<const float>();

  EXPECT_NEAR(out[0], 2.f, 1e-5);  // wo swaps v = {1, 2}
  EXPECT_NEAR(out[1], 1.f, 1e-5);
}

// wq = wk = 0 → all attention scores are 0 → uniform weights over the cache.
// (Zero vectors are RoPE-invariant, so no trig enters the expectations.)
// With wv = wo = I the output is the running mean of the inputs — verifying
// the KV cache accumulates across Forward calls and positions advance.
TEST(GroupQueryAttnBlockTest, UniformAttentionAveragesCachedValues) {
  ComputeEngine engine;
  GroupQueryAttnBlock block(
      engine, TinyGqaParam(AsMat(kZero2, 2, 2), AsMat(kZero2, 2, 2),
                           AsMat(kIdentity2, 2, 2), AsMat(kIdentity2, 2, 2)));

  const std::vector<float> x0 = {2.f, 0.f};
  std::span<const float> out0 = block.Forward(AsRow(x0)).As<const float>();
  EXPECT_NEAR(out0[0], 2.f, 1e-5);
  EXPECT_NEAR(out0[1], 0.f, 1e-5);

  const std::vector<float> x1 = {4.f, 2.f};
  std::span<const float> out1 = block.Forward(AsRow(x1)).As<const float>();
  EXPECT_NEAR(out1[0], 3.f, 1e-5);  // mean of v0 = {2,0}, v1 = {4,2}
  EXPECT_NEAR(out1[1], 1.f, 1e-5);
}

// Same setup, both tokens in one batched call. Row 0 must see only its own
// value (causal mask), row 1 the mean of both — identical to the sequential
// results above.
TEST(GroupQueryAttnBlockTest, UniformAttentionBatchedIsCausal) {
  ComputeEngine engine;
  GroupQueryAttnBlock block(
      engine, TinyGqaParam(AsMat(kZero2, 2, 2), AsMat(kZero2, 2, 2),
                           AsMat(kIdentity2, 2, 2), AsMat(kIdentity2, 2, 2)));

  const std::vector<float> x = {2.f, 0.f,
                                4.f, 2.f};
  MatrixView out_view = block.Forward(AsMat(x, 2, 2));
  ASSERT_EQ(out_view.shape[0], 2);
  std::span<const float> out = out_view.As<const float>();

  EXPECT_NEAR(out[0], 2.f, 1e-5);  // row 0: only v0
  EXPECT_NEAR(out[1], 0.f, 1e-5);
  EXPECT_NEAR(out[2], 3.f, 1e-5);  // row 1: mean(v0, v1)
  EXPECT_NEAR(out[3], 1.f, 1e-5);
}

// The prefill contract at block level: one Forward over [x0; x1; x2] on a
// fresh block must equal three single-token Forwards on another fresh block.
// Non-zero q/k make RoPE per-row positions and the causal mask load-bearing;
// a wrong per-row position or a missing mask changes rows 1 and 2.
TEST(GroupQueryAttnBlockTest, BatchedPrefillMatchesSequential) {
  ComputeEngine engine;
  const std::vector<float> wo = {1.f, 2.f, 3.f, 4.f};
  GroupQueryAttnBlock batched_block(engine, MixingGqaParam(wo));
  GroupQueryAttnBlock serial_block(engine, MixingGqaParam(wo));

  const std::vector<float> x = {1.f,   2.f,
                                -0.5f, 1.5f,
                                2.f,   -1.f};
  const std::vector<float> batched =
      ToVector(batched_block.Forward(AsMat(x, 3, 2)));
  ASSERT_EQ(batched.size(), 6u);

  for (int64_t t = 0; t < 3; ++t) {
    const std::vector<float> serial =
        ToVector(serial_block.Forward(AsMat(x, 3, 2).Slice(t, 1)));
    EXPECT_NEAR(batched[t * 2], serial[0], 1e-5) << "t=" << t;
    EXPECT_NEAR(batched[t * 2 + 1], serial[1], 1e-5) << "t=" << t;
  }
}

// Prefill followed by decode: after a 2-token batch, a single-token Forward
// must see both cached tokens at the right positions. Compared against a
// block fed all three tokens one at a time. Pins token_index_ advancing by T
// and the cache rows being written at the batch's positions.
TEST(GroupQueryAttnBlockTest, PrefillThenDecodeMatchesSequential) {
  ComputeEngine engine;
  const std::vector<float> wo = {1.f, 2.f, 3.f, 4.f};
  GroupQueryAttnBlock prefill_block(engine, MixingGqaParam(wo));
  GroupQueryAttnBlock serial_block(engine, MixingGqaParam(wo));

  const std::vector<float> x = {1.f,   2.f,
                                -0.5f, 1.5f,
                                2.f,   -1.f};
  prefill_block.Forward(AsMat(x, 3, 2).Top(2));
  const std::vector<float> decoded =
      ToVector(prefill_block.Forward(AsMat(x, 3, 2).Slice(2, 1)));

  std::vector<float> serial;
  for (int64_t t = 0; t < 3; ++t) {
    serial = ToVector(serial_block.Forward(AsMat(x, 3, 2).Slice(t, 1)));
  }

  ASSERT_EQ(decoded.size(), 2u);
  EXPECT_NEAR(decoded[0], serial[0], 1e-5);
  EXPECT_NEAR(decoded[1], serial[1], 1e-5);
}

// ── DecoderBlock ──────────────────────────────────────────────────────────────

DecoderBlock::Param TinyDecoderParam(MatrixView wo, MatrixView wdown) {
  DecoderBlock::Param param;
  param.embedding_size = 2;
  param.attn_norm = {AsVec(kOnes2), kEps};
  param.gqa = TinyGqaParam(AsMat(kIdentity2, 2, 2), AsMat(kIdentity2, 2, 2),
                           AsMat(kIdentity2, 2, 2), wo);
  param.ffn_norm = {AsVec(kOnes2), kEps};
  param.swiglu_ffn = {.wup = AsMat(kIdentity2, 2, 2),
                      .wgate = AsMat(kSwap2, 2, 2),
                      .wdown = wdown};
  return param;
}

// With wo = 0 and wdown = 0 both sub-blocks contribute nothing, so the two
// residual adds must reproduce the input exactly. Pins the pre-norm residual
// wiring: both adds use the UN-normed running value, not the norm outputs.
TEST(DecoderBlockTest, ZeroedSubBlocksGiveIdentity) {
  ComputeEngine engine;
  DecoderBlock block(engine,
                     TinyDecoderParam(AsMat(kZero2, 2, 2), AsMat(kZero2, 2, 2)));

  const std::vector<float> x = {3.f, -1.f};
  std::span<const float> out = block.Forward(AsRow(x)).As<const float>();

  EXPECT_FLOAT_EQ(out[0], 3.f);
  EXPECT_FLOAT_EQ(out[1], -1.f);
}

// DecoderBlock must equal the manual composition of its parts (independently
// constructed with the same params): attn_out = x + GQA(norm(x));
// out = attn_out + FFN(norm(attn_out)). Bit-identical, same kernel sequence.
TEST(DecoderBlockTest, MatchesManualComposition) {
  ComputeEngine engine;
  const std::vector<float> wdown = {1.f, 2.f, 3.f, 4.f};
  DecoderBlock block(engine, TinyDecoderParam(AsMat(kSwap2, 2, 2),
                                              AsMat(wdown, 2, 2)));

  // Manual composition with fresh instances of the same sub-blocks.
  RmsNormLayer attn_norm(engine, {AsVec(kOnes2), kEps});
  GroupQueryAttnBlock gqa(
      engine, TinyGqaParam(AsMat(kIdentity2, 2, 2), AsMat(kIdentity2, 2, 2),
                           AsMat(kIdentity2, 2, 2), AsMat(kSwap2, 2, 2)));
  RmsNormLayer ffn_norm(engine, {AsVec(kOnes2), kEps});
  SwiGluFfnBlock ffn(engine, {.wup = AsMat(kIdentity2, 2, 2),
                              .wgate = AsMat(kSwap2, 2, 2),
                              .wdown = AsMat(wdown, 2, 2)});

  const std::vector<float> x = {1.f, 2.f};
  std::span<const float> block_out = block.Forward(AsRow(x)).As<const float>();

  std::vector<float> attn_out(2), expected(2);
  MatrixView gqa_out = gqa.Forward(attn_norm.Forward(AsRow(x)));
  engine.Add(AsMutableMat(attn_out, 1, 2), AsRow(x), gqa_out);
  MatrixView ffn_out = ffn.Forward(ffn_norm.Forward(AsRow(attn_out)));
  engine.Add(AsMutableMat(expected, 1, 2), AsRow(attn_out), ffn_out);

  EXPECT_FLOAT_EQ(block_out[0], expected[0]);
  EXPECT_FLOAT_EQ(block_out[1], expected[1]);
}

// End-to-end prefill contract for a full decoder block: a batched forward
// over three tokens equals three sequential single-token forwards on an
// identically constructed block. Every prefill path participates: per-row
// norm, batched projections, per-row RoPE, causal attention, residual adds.
TEST(DecoderBlockTest, BatchedPrefillMatchesSequential) {
  ComputeEngine engine;
  const std::vector<float> wdown = {1.f, 2.f, 3.f, 4.f};
  DecoderBlock batched_block(
      engine, TinyDecoderParam(AsMat(kSwap2, 2, 2), AsMat(wdown, 2, 2)));
  DecoderBlock serial_block(
      engine, TinyDecoderParam(AsMat(kSwap2, 2, 2), AsMat(wdown, 2, 2)));

  const std::vector<float> x = {1.f,   2.f,
                                -0.5f, 1.5f,
                                2.f,   -1.f};
  MatrixView batched_view = batched_block.Forward(AsMat(x, 3, 2));
  ASSERT_EQ(batched_view.shape[0], 3);
  ASSERT_EQ(batched_view.shape[1], 2);
  const std::vector<float> batched = ToVector(batched_view);

  for (int64_t t = 0; t < 3; ++t) {
    const std::vector<float> serial =
        ToVector(serial_block.Forward(AsMat(x, 3, 2).Slice(t, 1)));
    EXPECT_NEAR(batched[t * 2], serial[0], 1e-5) << "t=" << t;
    EXPECT_NEAR(batched[t * 2 + 1], serial[1], 1e-5) << "t=" << t;
  }
}

// Prefill two tokens, then decode the third; the decode output must match a
// block that saw all three tokens sequentially. This is the chat flow: prompt
// prefill followed by token-by-token generation over the same cache.
TEST(DecoderBlockTest, PrefillThenDecodeMatchesSequential) {
  ComputeEngine engine;
  const std::vector<float> wdown = {1.f, 2.f, 3.f, 4.f};
  DecoderBlock prefill_block(
      engine, TinyDecoderParam(AsMat(kSwap2, 2, 2), AsMat(wdown, 2, 2)));
  DecoderBlock serial_block(
      engine, TinyDecoderParam(AsMat(kSwap2, 2, 2), AsMat(wdown, 2, 2)));

  const std::vector<float> x = {1.f,   2.f,
                                -0.5f, 1.5f,
                                2.f,   -1.f};
  prefill_block.Forward(AsMat(x, 3, 2).Top(2));
  const std::vector<float> decoded =
      ToVector(prefill_block.Forward(AsMat(x, 3, 2).Slice(2, 1)));

  std::vector<float> serial;
  for (int64_t t = 0; t < 3; ++t) {
    serial = ToVector(serial_block.Forward(AsMat(x, 3, 2).Slice(t, 1)));
  }

  ASSERT_EQ(decoded.size(), 2u);
  EXPECT_NEAR(decoded[0], serial[0], 1e-5);
  EXPECT_NEAR(decoded[1], serial[1], 1e-5);
}

}  // namespace
}  // namespace tlm
