#pragma once

#include <string>
#include <optional>

#include "model.h"

namespace tlm {

class ModelLoader {
 public:
  explicit ModelLoader(const std::string& model_path);
  ~ModelLoader();

  std::optional<Model> Load();
};

}  // namespace tlm
