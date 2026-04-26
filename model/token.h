#pragma once

#include <cstdint>

namespace tlm {

struct Token {
  int32_t id;

  static const Token BOS;
  static const Token EOS;
  static const Token INVALID;

  bool operator==(const Token& other) const { return id == other.id; }
  bool operator!=(const Token& other) const { return id != other.id; }
};

inline const Token Token::BOS = {1};
inline const Token Token::EOS = {2};
inline const Token Token::INVALID = {-1};

}  // namespace tlm
