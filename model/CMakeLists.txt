add_library(model OBJECT
  model.cc
  model_loader.cc
)

target_link_libraries(model
  PUBLIC base
  PRIVATE gguf glog::glog
)
