#include "base/mmap_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <glog/logging.h>

#include "base/scoped_fd.h"

namespace tlm {

void MmapFileDeleter::operator()(std::byte* ptr) const {
  if (!ptr) {
    return;
  }

  munmap(ptr, length);
}

ScopedMmapFile OpenMmapFile(const std::string& path) {
  ScopedFd fd(open(path.c_str(), O_RDONLY));
  if (fd.Get() < 0) {
    LOG(ERROR) << "Failed to open file at " << path << " ret " << fd.Get();
    return ScopedMmapFile();
  }

  struct stat sb;
  fstat(fd.Get(), &sb);
  size_t file_len = sb.st_size;

  void* ptr = mmap(nullptr, file_len, PROT_READ, MAP_PRIVATE, fd.Get(), 0);
  if (ptr == MAP_FAILED) {
    LOG(ERROR) << "Fail to mmap";
    return ScopedMmapFile();
  }

  MmapFileDeleter deleter;
  deleter.length = file_len;

  return ScopedMmapFile(reinterpret_cast<std::byte*>(ptr), deleter);
}

size_t MmapFileGetSize(const ScopedMmapFile& file) {
  return file.get_deleter().length;
}
}  // namespace tlm
