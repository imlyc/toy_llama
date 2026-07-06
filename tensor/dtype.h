#pragma once

#include <iostream>
#include <string>

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

int GetDTypeBlockBytes(DType type);
int GetDTypeBlockElements(DType type);

std::ostream& operator<<(std::ostream& oss, DType type);

}  // namespace tlm
