#include "model_loader.h"

#include <glog/logging.h>

#include "base/mmap_file.h"
#include "gguf/gguf_parser.h"

namespace tlm {
ModelLoader::ModelLoader() = default;
ModelLoader::~ModelLoader() = default;

std::unique_ptr<Model> ModelLoader::Load(const std::string& model_path) {
  ScopedMmapFile model_file = OpenMmapFile(model_path);
  if (!model_file) {
    LOG(ERROR) << "Failed to open model file " << model_path;
    return nullptr;
  }

  GgufParser parser;
  if (!parser.Parse(model_file.get(), MmapFileGetSize(model_file))) {
    LOG(ERROR) << "Failed to parser gguf";
    return nullptr;
  }

  auto model = std::make_unique<Model>(std::move(model_file));
  return model;
}

}  // namespace tlm
