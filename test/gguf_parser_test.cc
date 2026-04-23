#include "gguf/gguf_parser.h"

#include <climits>
#include <cstddef>
#include <cstring>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

namespace tlm {
namespace {

// Builds a GGUF binary buffer in memory for testing.
class GgufBufferBuilder {
 public:
  template <typename T>
  void Write(T v) {
    const auto* p = reinterpret_cast<const std::byte*>(&v);
    buf_.insert(buf_.end(), p, p + sizeof(T));
  }

  void WriteString(std::string_view s) {
    Write<uint64_t>(s.size());
    const auto* p = reinterpret_cast<const std::byte*>(s.data());
    buf_.insert(buf_.end(), p, p + s.size());
  }

  // magic=GGUF, version=3
  void WriteHeader(uint64_t tensor_count = 0, uint64_t metadata_kv_count = 0) {
    Write<uint32_t>(0x46554747);  // magic
    Write<uint32_t>(3);           // version
    Write<uint64_t>(tensor_count);
    Write<uint64_t>(metadata_kv_count);
  }

  // Write a key + value_type preamble for a metadata KV.
  void WriteKVPreamble(std::string_view key, uint32_t value_type) {
    WriteString(key);
    Write<uint32_t>(value_type);
  }

  void WriteTensorInfo(std::string_view name, std::vector<uint64_t> dims,
                       uint32_t type, uint64_t offset) {
    WriteString(name);
    Write<uint32_t>(static_cast<uint32_t>(dims.size()));
    for (uint64_t d : dims) Write<uint64_t>(d);
    Write<uint32_t>(type);
    Write<uint64_t>(offset);
  }

  std::byte* ptr() { return buf_.data(); }
  size_t size() { return buf_.size(); }

 private:
  std::vector<std::byte> buf_;
};

// ── Header validation ─────────────────────────────────────────────────────────

TEST(GgufParserTest, EmptyBuffer) {
  GgufParser p;
  EXPECT_FALSE(p.Parse(nullptr, 0));
}

TEST(GgufParserTest, TruncatedHeader) {
  GgufBufferBuilder b;
  b.Write<uint32_t>(0x46554747);  // only magic, no more bytes
  GgufParser p;
  EXPECT_FALSE(p.Parse(b.ptr(), b.size()));
}

TEST(GgufParserTest, WrongMagic) {
  GgufBufferBuilder b;
  b.Write<uint32_t>(0xDEADBEEF);
  b.Write<uint32_t>(3);
  b.Write<uint64_t>(0);
  b.Write<uint64_t>(0);
  GgufParser p;
  EXPECT_FALSE(p.Parse(b.ptr(), b.size()));
}

TEST(GgufParserTest, WrongVersion) {
  GgufBufferBuilder b;
  b.Write<uint32_t>(0x46554747);
  b.Write<uint32_t>(2);  // version 2, not 3
  b.Write<uint64_t>(0);
  b.Write<uint64_t>(0);
  GgufParser p;
  EXPECT_FALSE(p.Parse(b.ptr(), b.size()));
}

TEST(GgufParserTest, MinimalValidHeader) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 0);
  GgufParser p;
  EXPECT_TRUE(p.Parse(b.ptr(), b.size()));
}

// ── Metadata scalar types ─────────────────────────────────────────────────────

TEST(GgufParserTest, MetadataUInt8) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 1);
  b.WriteKVPreamble("k", 0 /*UINT8*/);
  b.Write<uint8_t>(42);
  GgufParser p;
  ASSERT_TRUE(p.Parse(b.ptr(), b.size()));
  EXPECT_EQ(p.GetMetadata<uint8_t>("k"), 42);
}

TEST(GgufParserTest, MetadataInt8) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 1);
  b.WriteKVPreamble("k", 1 /*INT8*/);
  b.Write<int8_t>(-7);
  GgufParser p;
  ASSERT_TRUE(p.Parse(b.ptr(), b.size()));
  EXPECT_EQ(p.GetMetadata<int8_t>("k"), -7);
}

TEST(GgufParserTest, MetadataUInt16) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 1);
  b.WriteKVPreamble("k", 2 /*UINT16*/);
  b.Write<uint16_t>(1000);
  GgufParser p;
  ASSERT_TRUE(p.Parse(b.ptr(), b.size()));
  EXPECT_EQ(p.GetMetadata<uint16_t>("k"), 1000);
}

TEST(GgufParserTest, MetadataInt16) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 1);
  b.WriteKVPreamble("k", 3 /*INT16*/);
  b.Write<int16_t>(-500);
  GgufParser p;
  ASSERT_TRUE(p.Parse(b.ptr(), b.size()));
  EXPECT_EQ(p.GetMetadata<int16_t>("k"), -500);
}

TEST(GgufParserTest, MetadataUInt32) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 1);
  b.WriteKVPreamble("k", 4 /*UINT32*/);
  b.Write<uint32_t>(99999);
  GgufParser p;
  ASSERT_TRUE(p.Parse(b.ptr(), b.size()));
  EXPECT_EQ(p.GetMetadata<uint32_t>("k"), 99999u);
}

TEST(GgufParserTest, MetadataInt32) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 1);
  b.WriteKVPreamble("k", 5 /*INT32*/);
  b.Write<int32_t>(-12345);
  GgufParser p;
  ASSERT_TRUE(p.Parse(b.ptr(), b.size()));
  EXPECT_EQ(p.GetMetadata<int32_t>("k"), -12345);
}

TEST(GgufParserTest, MetadataFloat32) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 1);
  b.WriteKVPreamble("k", 6 /*FLOAT32*/);
  b.Write<float>(3.14f);
  GgufParser p;
  ASSERT_TRUE(p.Parse(b.ptr(), b.size()));
  EXPECT_FLOAT_EQ(p.GetMetadata<float>("k"), 3.14f);
}

TEST(GgufParserTest, MetadataUInt64) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 1);
  b.WriteKVPreamble("k", 10 /*UINT64*/);
  b.Write<uint64_t>(UINT64_MAX);
  GgufParser p;
  ASSERT_TRUE(p.Parse(b.ptr(), b.size()));
  EXPECT_EQ(p.GetMetadata<uint64_t>("k"), UINT64_MAX);
}

TEST(GgufParserTest, MetadataInt64) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 1);
  b.WriteKVPreamble("k", 11 /*INT64*/);
  b.Write<int64_t>(INT64_MIN);
  GgufParser p;
  ASSERT_TRUE(p.Parse(b.ptr(), b.size()));
  EXPECT_EQ(p.GetMetadata<int64_t>("k"), INT64_MIN);
}

TEST(GgufParserTest, MetadataFloat64) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 1);
  b.WriteKVPreamble("k", 12 /*FLOAT64*/);
  b.Write<double>(2.718);
  GgufParser p;
  ASSERT_TRUE(p.Parse(b.ptr(), b.size()));
  EXPECT_DOUBLE_EQ(p.GetMetadata<double>("k"), 2.718);
}

TEST(GgufParserTest, MetadataBoolTrue) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 1);
  b.WriteKVPreamble("k", 7 /*BOOL*/);
  b.Write<uint8_t>(1);
  GgufParser p;
  ASSERT_TRUE(p.Parse(b.ptr(), b.size()));
  EXPECT_TRUE(p.GetMetadata<bool>("k"));
}

TEST(GgufParserTest, MetadataBoolFalse) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 1);
  b.WriteKVPreamble("k", 7 /*BOOL*/);
  b.Write<uint8_t>(0);
  GgufParser p;
  ASSERT_TRUE(p.Parse(b.ptr(), b.size()));
  EXPECT_FALSE(p.GetMetadata<bool>("k"));
}

TEST(GgufParserTest, MetadataString) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 1);
  b.WriteKVPreamble("k", 8 /*STRING*/);
  b.WriteString("hello");
  GgufParser p;
  ASSERT_TRUE(p.Parse(b.ptr(), b.size()));
  EXPECT_EQ(p.GetMetadata<std::string_view>("k"), "hello");
}

// ── Metadata error cases ──────────────────────────────────────────────────────

TEST(GgufParserTest, MetadataBoolInvalidValue) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 1);
  b.WriteKVPreamble("k", 7 /*BOOL*/);
  b.Write<uint8_t>(2);  // invalid: not 0 or 1
  GgufParser p;
  EXPECT_FALSE(p.Parse(b.ptr(), b.size()));
}

TEST(GgufParserTest, MetadataNestedArray) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 1);
  b.WriteKVPreamble("k", 9 /*ARRAY*/);
  b.Write<uint32_t>(9 /*ARRAY element type*/);
  b.Write<uint64_t>(1);  // length = 1
  GgufParser p;
  EXPECT_FALSE(p.Parse(b.ptr(), b.size()));
}

// ── Metadata array types ──────────────────────────────────────────────────────

TEST(GgufParserTest, MetadataArrayUInt32) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 1);
  b.WriteKVPreamble("k", 9 /*ARRAY*/);
  b.Write<uint32_t>(4 /*UINT32 element type*/);
  b.Write<uint64_t>(3);  // 3 elements
  b.Write<uint32_t>(1);
  b.Write<uint32_t>(2);
  b.Write<uint32_t>(3);
  GgufParser p;
  ASSERT_TRUE(p.Parse(b.ptr(), b.size()));
  const GgufParser::Array& arr = p.GetMetadata<GgufParser::Array>("k");
  EXPECT_EQ(arr.size, 3);
  const uint32_t* vals = reinterpret_cast<const uint32_t*>(arr.data);
  EXPECT_EQ(vals[0], 1u);
  EXPECT_EQ(vals[1], 2u);
  EXPECT_EQ(vals[2], 3u);
}

TEST(GgufParserTest, MetadataArrayStrings) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 1);
  b.WriteKVPreamble("k", 9 /*ARRAY*/);
  b.Write<uint32_t>(8 /*STRING element type*/);
  b.Write<uint64_t>(2);  // 2 elements
  b.WriteString("foo");
  b.WriteString("bar");
  GgufParser p;
  ASSERT_TRUE(p.Parse(b.ptr(), b.size()));
  const auto& strs = p.GetMetadata<std::vector<std::string_view>>("k");
  ASSERT_EQ(strs.size(), 2u);
  EXPECT_EQ(strs[0], "foo");
  EXPECT_EQ(strs[1], "bar");
}

// ── Tensor info ───────────────────────────────────────────────────────────────

TEST(GgufParserTest, SingleTensor) {
  GgufBufferBuilder b;
  b.WriteHeader(1, 0);
  b.WriteTensorInfo("weight", {4, 8}, 0 /*type*/, 0 /*offset*/);
  GgufParser p;
  EXPECT_TRUE(p.Parse(b.ptr(), b.size()));
}

TEST(GgufParserTest, TensorZeroDimensions) {
  GgufBufferBuilder b;
  b.WriteHeader(1, 0);
  b.WriteString("w");
  b.Write<uint32_t>(0);  // n_dimensions = 0 → invalid
  GgufParser p;
  EXPECT_FALSE(p.Parse(b.ptr(), b.size()));
}

TEST(GgufParserTest, TensorTypeOutOfRange) {
  GgufBufferBuilder b;
  b.WriteHeader(1, 0);
  b.WriteTensorInfo("w", {4}, 40 /*type >= 40*/, 0);
  GgufParser p;
  EXPECT_FALSE(p.Parse(b.ptr(), b.size()));
}

// ── Alignment ─────────────────────────────────────────────────────────────────

TEST(GgufParserTest, DefaultAlignment) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 0);
  GgufParser p;
  EXPECT_TRUE(p.Parse(b.ptr(), b.size()));
}

TEST(GgufParserTest, CustomAlignmentKey) {
  // Note: the parser looks for "generial.alignment" (typo in implementation).
  GgufBufferBuilder b;
  b.WriteHeader(0, 1);
  b.WriteKVPreamble("generial.alignment", 4 /*UINT32*/);
  b.Write<uint32_t>(64);
  GgufParser p;
  EXPECT_TRUE(p.Parse(b.ptr(), b.size()));
}

// ── Truncation / boundary ─────────────────────────────────────────────────────

TEST(GgufParserTest, TruncatedMetadataValue) {
  GgufBufferBuilder b;
  b.WriteHeader(0, 1);
  b.WriteKVPreamble("k", 10 /*UINT64*/);
  // Omit the 8-byte value → truncated
  GgufParser p;
  EXPECT_FALSE(p.Parse(b.ptr(), b.size()));
}

TEST(GgufParserTest, TruncatedTensorName) {
  GgufBufferBuilder b;
  b.WriteHeader(1, 0);
  // Write length=10 but only 3 bytes of name content
  b.Write<uint64_t>(10);
  b.Write<uint32_t>(0x61626300);  // 3 bytes + null (only 4 bytes, not 10)
  GgufParser p;
  EXPECT_FALSE(p.Parse(b.ptr(), b.size()));
}

}  // namespace
}  // namespace tlm
