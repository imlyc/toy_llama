#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include "base/mmap_file.h"

namespace tlm {
class GgufParser;

class Model {
 public:
  Model(ScopedMmapFile file, std::unique_ptr<GgufParser> parser);
  ~Model();

  const std::vector<std::string_view>& GetTokenizerMerges() const;
  const std::vector<std::string_view>& GetTokenizerTokens() const;

 private:
  ScopedMmapFile model_file_;
  std::unique_ptr<GgufParser> parser_;
};

}  // namespace tlm
