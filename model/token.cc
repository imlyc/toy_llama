#include "model/token.h"

namespace tlm {

std::ostream& operator<<(std::ostream& oss, Token token) {
  oss << "Token{" << token.id << "}";
  return oss;
}

}  // namespace tlm
