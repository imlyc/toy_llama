#pragma once

#include <utility>

#include <glog/logging.h>

namespace tlm {

template <typename To, typename From>
To CheckedCast(From v) {
  CHECK(std::in_range<To>(v));
  return static_cast<To>(v);
}
}  // namespace
