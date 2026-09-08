#include <cmath>
#include <cstddef>
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
  VectorView out = layer.Forward(AsVec(x));

  ASSERT_EQ(out.shape[0], 2);
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
  std::span<const float> first = layer.Forward(AsVec(x)).As<const float>();
  const float f0 = first[0], f1 = first[1];
  std::span<const float> second = layer.Forward(AsVec(x)).As<const float>();

  EXPECT_FLOAT_EQ(second[0], f0);
  EXPECT_FLOAT_EQ(second[1], f1);
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
  std::span<const float> out = block.Forward(AsVec(x)).As<const float>();

  // gate after SwiGluMul: {1.7615942, 1.4621172}
  EXPECT_NEAR(out[0], 4.6858287f, 1e-4);   // 1.7615942 + 2 * 1.4621172
  EXPECT_NEAR(out[1], 11.133251f, 1e-4);   // 3 * 1.7615942 + 4 * 1.4621172
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

// First token: position 0 makes RoPE the identity, and softmax over a single
// cached key is 1 regardless of q/k. So output = wo @ (wv @ x).
TEST(GroupQueryAttnBlockTest, SingleTokenPassesValueThroughWo) {
  ComputeEngine engine;
  GroupQueryAttnBlock block(
      engine, TinyGqaParam(AsMat(kIdentity2, 2, 2), AsMat(kIdentity2, 2, 2),
                           AsMat(kIdentity2, 2, 2), AsMat(kSwap2, 2, 2)));

  const std::vector<float> x = {1.f, 2.f};
  std::span<const float> out = block.Forward(AsVec(x)).As<const float>();

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
  std::span<const float> out0 = block.Forward(AsVec(x0)).As<const float>();
  EXPECT_NEAR(out0[0], 2.f, 1e-5);
  EXPECT_NEAR(out0[1], 0.f, 1e-5);

  const std::vector<float> x1 = {4.f, 2.f};
  std::span<const float> out1 = block.Forward(AsVec(x1)).As<const float>();
  EXPECT_NEAR(out1[0], 3.f, 1e-5);  // mean of v0 = {2,0}, v1 = {4,2}
  EXPECT_NEAR(out1[1], 1.f, 1e-5);
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
  std::span<const float> out = block.Forward(AsVec(x)).As<const float>();

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
  std::span<const float> block_out = block.Forward(AsVec(x)).As<const float>();

  std::vector<float> attn_out(2), expected(2);
  VectorView gqa_out = gqa.Forward(attn_norm.Forward(AsVec(x)));
  engine.Add(AsMutableVec(attn_out), AsVec(x), gqa_out);
  VectorView ffn_out = ffn.Forward(ffn_norm.Forward(AsVec(attn_out)));
  engine.Add(AsMutableVec(expected), AsVec(attn_out), ffn_out);

  EXPECT_FLOAT_EQ(block_out[0], expected[0]);
  EXPECT_FLOAT_EQ(block_out[1], expected[1]);
}

}  // namespace
}  // namespace tlm
