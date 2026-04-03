#include "model_loader.h"

namespace tlm {

ModelLoader::ModelLoader(const std::string& model_path) {}
ModelLoader::~ModelLoader() = default;

std::optional<Model> ModelLoader::Load() {
  return std::nullopt;
}

}  // namespace tlm
