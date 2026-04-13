#include "base/scoped_fd.h"

#include <unistd.h>

namespace tlm {

ScopedFd::ScopedFd(): ScopedFd(-1) {}
ScopedFd::ScopedFd(int fd): fd_(fd) {}
ScopedFd::~ScopedFd() {
  Close();
}

void ScopedFd::Set(int fd) {
  Close();
  fd_ = fd;
}

int ScopedFd::Get() const {
  return fd_;
}

void ScopedFd::Close() {
  if (fd_ < 0) {
    return;
  }
  close(fd_);
  fd_ = -1;
}
}  // namespace tlm
