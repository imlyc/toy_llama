#pragma once

#include <iostream>
#include <string>

#include "base/float16.h"

namespace tlm {

enum class DType : int8_t {
  F32 = 0,
  Q8_0 = 1,
};

struct DTypeTrait {
  int block_bytes = 0;
  int block_elements = 0;
  const char* name = nullptr;
};

struct __attribute__((packed)) BlockQ8_0 {
  Float16 scale = 1.;
  int8_t data[32] = {0};
};
static_assert(sizeof(BlockQ8_0) ==
                  sizeof(BlockQ8_0::scale) + sizeof(BlockQ8_0::data),
              "BlockQ8_0 should be packed");

int GetDTypeBlockBytes(DType type);
int GetDTypeBlockElements(DType type);

std::ostream& operator<<(std::ostream& oss, DType type);

}  // namespace tlm
