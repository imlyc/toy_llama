#pragma once

#include <cstdint>

namespace tlm {

struct Token {
  int32_t id;

  static const Token BOS;
  static const Token EOS;

  bool operator==(const Token& other) const { return id == other.id; }
  bool operator!=(const Token& other) const { return id != other.id; }
};

inline const Token Token::BOS = {1};
inline const Token Token::EOS = {2};

}  // namespace tlm
