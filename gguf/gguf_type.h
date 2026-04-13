#pragma once

#include <cstdint>

namespace tlm {

struct GgufHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_kv_count;
};

enum class GgufMetadataValueType : uint32_t {
  // The value is a 8-bit unsigned integer.
  UINT8 = 0,
  // The value is a 8-bit signed integer.
  INT8 = 1,
  // The value is a 16-bit unsigned little-endian integer.
  UINT16 = 2,
  // The value is a 16-bit signed little-endian integer.
  INT16 = 3,
  // The value is a 32-bit unsigned little-endian integer.
  UINT32 = 4,
  // The value is a 32-bit signed little-endian integer.
  INT32 = 5,
  // The value is a 32-bit IEEE754 floating point number.
  FLOAT32 = 6,
  // The value is a boolean.
  // 1-byte value where 0 is false and 1 is true.
  // Anything else is invalid, and should be treated as either the model being
  // invalid or the reader being buggy.
  BOOL = 7,
  // The value is a UTF-8 non-null-terminated string, with length prepended.
  STRING = 8,
  // The value is an array of other values, with the length and type prepended.
  ///
  // Arrays can be nested, and the length of the array is the number of elements
  // in the array, not the number of bytes.
  ARRAY = 9,
  // The value is a 64-bit unsigned little-endian integer.
  UINT64 = 10,
  // The value is a 64-bit signed little-endian integer.
  INT64 = 11,
  // The value is a 64-bit IEEE754 floating point number.
  FLOAT64 = 12,
};

}  // namespace tlm
