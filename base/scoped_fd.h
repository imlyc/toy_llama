#pragma once

namespace tlm {

class ScopedFd {
 public:
  ScopedFd();
  explicit ScopedFd(int fd);
  ~ScopedFd();

  void Set(int fd);
  int Get() const;

 private:
  void Close();

  int fd_;
};

}  // namespace tlm
