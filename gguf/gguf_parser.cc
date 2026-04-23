#include "gguf/gguf_parser.h"

#include <glog/logging.h>

namespace tlm {
#define RET_CHECK(expr)                                      \
    do {                                                     \
        if (!(expr)) {                                       \
            LOG(ERROR) << "RET_CHECK failed: " << #expr;     \
            return false;                                    \
        }                                                    \
    } while (0)

GgufParser::GgufParser() = default;
GgufParser::~GgufParser() = default;

bool GgufParser::Parse(std::byte* mem, size_t size) {
  mem_ = reinterpret_cast<std::byte*>(mem);
  size_ = static_cast<int64_t>(size);
  next_ = 0;

  RET_CHECK(ParseHeader());

  LOG(INFO) << "version " << header_.version << " tensor_count "
            << header_.tensor_count << " metadata_kv_count "
            << header_.metadata_kv_count;

  RET_CHECK(ParseMetadata());
  RET_CHECK(ParseTensorInfos());

  LOG(INFO) << "Alignment = " << alignment_;
  return true;
}

bool GgufParser::ParseHeader() {
  RET_CHECK(ReadType<uint32_t>(&header_.magic));
  // GGUF: 0x47 0x47 0x55 0x46
  RET_CHECK(header_.magic == 0x46554747);
  RET_CHECK(ReadType<uint32_t>(&header_.version));
  RET_CHECK(header_.version == 3);
  RET_CHECK(ReadType<uint64_t>(&header_.tensor_count));
  RET_CHECK(ReadType<uint64_t>(&header_.metadata_kv_count));

  return true;
}

bool GgufParser::ParseMetadata() {
  for (int i = 0; i < header_.metadata_kv_count; i++) {
    RET_CHECK(ParseMetadataKV());
  }
  auto it = metadata_.find("generial.alignment");
  if (it != metadata_.end()) {
    RET_CHECK(std::holds_alternative<uint32_t>(it->second));
    alignment_ = std::get<uint32_t>(it->second);
  }
  return true;
}

bool GgufParser::ParseMetadataKV() {
  std::string_view key;
  RET_CHECK(ReadString(&key));
  uint32_t value_type;
  RET_CHECK(ReadType<uint32_t>(&value_type));

  if (static_cast<GgufMetadataValueType>(value_type) !=
      GgufMetadataValueType::ARRAY) {
    LOG(INFO) << "meatadata " << key << " type " << value_type;
  }

  switch (static_cast<GgufMetadataValueType>(value_type)) {
    case GgufMetadataValueType::UINT8:
      RET_CHECK(ParseMetadataValue<uint8_t>(key));
      break;
    case GgufMetadataValueType::INT8:
      RET_CHECK(ParseMetadataValue<int8_t>(key));
      break;
    case GgufMetadataValueType::UINT16:
      RET_CHECK(ParseMetadataValue<uint16_t>(key));
      break;
    case GgufMetadataValueType::INT16:
      RET_CHECK(ParseMetadataValue<int16_t>(key));
      break;
    case GgufMetadataValueType::UINT32:
      RET_CHECK(ParseMetadataValue<uint32_t>(key));
      break;
    case GgufMetadataValueType::INT32:
      RET_CHECK(ParseMetadataValue<int32_t>(key));
      break;
    case GgufMetadataValueType::FLOAT32:
      RET_CHECK(ParseMetadataValue<float>(key));
      break;
    case GgufMetadataValueType::BOOL:
      RET_CHECK(ParseMetadataValueBool(key));
      break;
    case GgufMetadataValueType::STRING:
      RET_CHECK(ParseMetadataValueString(key));
      break;
    case GgufMetadataValueType::ARRAY:
      RET_CHECK(ParseMetadataValueArray(key));
      break;
    case GgufMetadataValueType::UINT64:
      RET_CHECK(ParseMetadataValue<uint64_t>(key));
      break;
    case GgufMetadataValueType::INT64:
      RET_CHECK(ParseMetadataValue<int64_t>(key));
      break;
    case GgufMetadataValueType::FLOAT64:
      RET_CHECK(ParseMetadataValue<double>(key));
      break;
  }

  return true;
}

template <typename T>
bool GgufParser::ParseMetadataValue(std::string_view key) {
  T raw_value;
  RET_CHECK(ReadType<T>(&raw_value));
  Value value(raw_value);
  metadata_.emplace(key, std::move(value));
  return true;
}

bool GgufParser::ParseMetadataValueBool(std::string_view key) {
  uint8_t raw_value;
  RET_CHECK(ReadType<uint8_t>(&raw_value));
  RET_CHECK(raw_value == 1 || raw_value == 0);
  Value value(raw_value == 1);
  metadata_.emplace(key, std::move(value));
  return true;
}

bool GgufParser::ParseMetadataValueString(std::string_view key) {
  std::string_view str;
  RET_CHECK(ReadString(&str));
  Value value(std::move(str));
  metadata_.emplace(key, std::move(value));
  return true;
}

bool GgufParser::ParseMetadataValueArray(std::string_view key) {
  Array array;

  uint32_t value_type;
  RET_CHECK(ReadType<uint32_t>(&value_type));
  uint64_t len;
  RET_CHECK(ReadType<uint64_t>(&len));

  array.size = len;

  LOG(INFO) << "metadata key " << key << " type array value type "
            << value_type;

  switch (static_cast<GgufMetadataValueType>(value_type)) {
    case GgufMetadataValueType::UINT8:
      RET_CHECK(ReadBlob<uint8_t>(&array.data, len));
      break;
    case GgufMetadataValueType::INT8:
      RET_CHECK(ReadBlob<int8_t>(&array.data, len));
      break;
    case GgufMetadataValueType::UINT16:
      RET_CHECK(ReadBlob<uint16_t>(&array.data, len));
      break;
    case GgufMetadataValueType::INT16:
      RET_CHECK(ReadBlob<int16_t>(&array.data, len));
      break;
    case GgufMetadataValueType::UINT32:
      RET_CHECK(ReadBlob<uint32_t>(&array.data, len));
      break;
    case GgufMetadataValueType::INT32:
      RET_CHECK(ReadBlob<int32_t>(&array.data, len));
      break;
    case GgufMetadataValueType::FLOAT32:
      RET_CHECK(ReadBlob<float>(&array.data, len));
      break;
    case GgufMetadataValueType::BOOL:
      RET_CHECK(ReadBlob<bool>(&array.data, len));
      break;
    case GgufMetadataValueType::UINT64:
      RET_CHECK(ReadBlob<uint64_t>(&array.data, len));
      break;
    case GgufMetadataValueType::INT64:
      RET_CHECK(ReadBlob<int64_t>(&array.data, len));
      break;
    case GgufMetadataValueType::FLOAT64:
      RET_CHECK(ReadBlob<double>(&array.data, len));
      break;

    case GgufMetadataValueType::STRING:
      for (size_t i = 0; i < len; i++) {
        std::string_view str;
        RET_CHECK(ReadString(&str));
        array.strings.emplace_back(str);
      }
      break;
    case GgufMetadataValueType::ARRAY:
      LOG(ERROR) << "nested arrary not supported: " << key;
      return false;
  }

  Value value(std::move(array));
  metadata_.emplace(key, std::move(value));
  return true;
}

bool GgufParser::ParseTensorInfos() {
  uint64_t tensor_data_size = 0;
  for (int i = 0; i < header_.tensor_count; i++) {
    TensorInfo info;
    RET_CHECK(ReadString(&info.name));

    uint32_t n_dimensions;
    RET_CHECK(ReadType<uint32_t>(&n_dimensions));
    RET_CHECK(n_dimensions > 0);

    std::byte* dimensions_ptr;
    RET_CHECK(ReadBlob<uint64_t>(&dimensions_ptr, n_dimensions));
    uint64_t* dimensions_u64_ptr = reinterpret_cast<uint64_t*>(dimensions_ptr);
    info.dimensions.insert(info.dimensions.end(), dimensions_u64_ptr,
                           dimensions_u64_ptr + n_dimensions);

    RET_CHECK(ReadType<uint32_t>(&info.type));
    RET_CHECK(info.type < 40);

    RET_CHECK(ReadType<uint64_t>(&info.offset));

    tensor_infos_.emplace(info.name, std::move(info));
  }

  // Alignment
  tensor_data_ = mem_ + ((next_ + alignment_ - 1) & ~(alignment_ - 1));

  // TODO: Sanity check.
  return true;
}

template <typename T>
bool GgufParser::ReadType(T* result) {
  CHECK(result);
  if (size_ - next_ < sizeof(T)) {
    return false;
  }
  memcpy(result, mem_ + next_, sizeof(T));
  next_ += sizeof(T);
  return true;
}

template <typename T>
bool GgufParser::ReadBlob(std::byte** result, int64_t size) {
  CHECK(result);
  if (size_ - next_ < sizeof(T) * size) {
    return false;
  }
  *result = mem_ + next_;
  next_ += sizeof(T) * size;
  return true;
}

bool GgufParser::ReadString(std::string_view* result) {
  CHECK(result);
  uint64_t len;
  RET_CHECK(ReadType<uint64_t>(&len));
  if (size_ - next_ < len) {
    return false;
  }

  *result = std::string_view(reinterpret_cast<char*>(mem_) + next_, len);
  next_ += len;
  return true;
}

}  // namespace tlm
