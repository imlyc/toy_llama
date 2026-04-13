#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace tlm {

struct MmapFileDeleter {
  size_t length = 0;
  void operator()(std::byte* ptr) const;
};

using ScopedMmapFile = std::unique_ptr<std::byte, MmapFileDeleter>;

ScopedMmapFile OpenMmapFile(const std::string& path);
size_t MmapFileGetSize(const ScopedMmapFile& file);

}  // namespace tlm
