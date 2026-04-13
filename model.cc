#include "model.h"

namespace tlm {

Model::Model(ScopedMmapFile file) : model_file_(std::move(file)) {}
Model::~Model() = default;

}  // namespace tlm
