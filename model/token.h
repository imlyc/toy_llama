#pragma once

#include <cstdint>

namespace tlm {

struct Token {
  int32_t id;

  static const Token INVALID;

  bool operator==(const Token& other) const { return id == other.id; }
  bool operator!=(const Token& other) const { return id != other.id; }
};

inline const Token Token::INVALID = {-1};

}  // namespace tlm
