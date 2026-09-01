#include "tensor/dtype.h"

#include <cstdint>

#include "base/float16.h"

namespace tlm {
namespace {
constexpr DTypeTrait kDTypeTraits[] = {
    {4, 1, "f32"},
    {sizeof(BlockQ8_0), sizeof(BlockQ8_0::data) / sizeof(int8_t), "q8_0"},
};
}  // namespace

int GetDTypeBlockBytes(DType type) {
  return kDTypeTraits[static_cast<int>(type)].block_bytes;
}

int GetDTypeBlockElements(DType type) {
  return kDTypeTraits[static_cast<int>(type)].block_elements;
}

std::ostream& operator<<(std::ostream& oss, DType type) {
  oss << kDTypeTraits[static_cast<int>(type)].name;
  return oss;
}
}  // namespace tlm
