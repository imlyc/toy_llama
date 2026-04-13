#pragma once

#include <memory>
#include <string>

#include "model.h"

namespace tlm {

class ModelLoader {
 public:
  ModelLoader();
  ~ModelLoader();

  std::unique_ptr<Model> Load(const std::string& model_path);
};

}  // namespace tlm
