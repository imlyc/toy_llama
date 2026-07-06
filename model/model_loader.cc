#include "model/model_loader.h"

#include <glog/logging.h>

#include "base/mmap_file.h"
#include "gguf/gguf_parser.h"

#include <cstdio>

namespace tlm {
ModelLoader::ModelLoader() = default;
ModelLoader::~ModelLoader() = default;

std::unique_ptr<Model> ModelLoader::Load(const std::string& model_path) {
  ScopedMmapFile model_file = OpenMmapFile(model_path);
  if (!model_file) {
    LOG(ERROR) << "Failed to open model file " << model_path;
    return nullptr;
  }

  auto parser = std::make_unique<GgufParser>();
  if (!parser->Parse(model_file.get(), MmapFileGetSize(model_file))) {
    LOG(ERROR) << "Failed to parser gguf";
    return nullptr;
  }

  auto model = std::make_unique<Model>(std::move(model_file), std::move(parser));
  return model;
}

}  // namespace tlm
