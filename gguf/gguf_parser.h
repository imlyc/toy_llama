#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include "gguf/gguf_type.h"

namespace tlm {

class GgufParser {
 public:
  GgufParser();
  ~GgufParser();

  struct Array;
  using Value = std::variant<uint8_t,
                             int8_t,
                             uint16_t,
                             int16_t,
                             uint32_t,
                             int32_t,
                             float,
                             bool,
                             std::string_view,
                             uint64_t,
                             int64_t,
                             double,
                             Array>;
  // Array is either a pointer to a continuious memory of primitive type, or an
  // array of string, or a nested array.
  struct Array {
    std::byte* data = nullptr;
    // size of the element, instead of the bytes.
    int64_t size = 0;
    std::vector<std::string_view> strings;
    std::vector<std::unique_ptr<Array>> arrays;
  };

  struct TensorInfo {
    std::string_view name;
    std::vector<uint64_t> dimensions;
    uint32_t type = 0;
    uint64_t offset = 0;
  };

  bool Parse(std::byte* mem, size_t size);

  template <typename T>
  const T& GetMetadata(std::string_view key) {
    return std::get<T>(metadata_[key]);
  }

  template <>
  const std::vector<std::string_view>& GetMetadata(std::string_view key) {
    const Array& array = GetMetadata<Array>(key);
    return array.strings;
  }

 private:
  template <typename T>
  bool ReadType(T* result);
  template <typename T>
  bool ReadBlob(std::byte** result, int64_t size);
  bool ReadString(std::string_view* result);

  bool ParseHeader();

  bool ParseMetadata();
  bool ParseMetadataKV();
  template <typename T>
  bool ParseMetadataValue(std::string_view key);
  bool ParseMetadataValueBool(std::string_view key);
  bool ParseMetadataValueString(std::string_view key);
  bool ParseMetadataValueArray(std::string_view key);

  bool ParseTensorInfos();

  std::byte* mem_ = nullptr;
  int64_t size_ = 0;
  int64_t next_ = 0;
  std::byte* tensor_data_ = nullptr;

  GgufHeader header_;
  uint64_t alignment_ = 32;
  std::unordered_map<std::string_view, Value> metadata_;
  std::unordered_map<std::string_view, TensorInfo> tensor_infos_;
};

}  // namespace tlm
