#pragma once

#include "base/mmap_file.h"

namespace tlm {

class Model {
 public:
  explicit Model(ScopedMmapFile file);
  ~Model();

 private:
  ScopedMmapFile model_file_;
};

}  // namespace tlm
