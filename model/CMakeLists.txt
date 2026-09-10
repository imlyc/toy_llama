add_library(model OBJECT
  model.cc
  model_loader.cc
  token.cc
)

target_link_libraries(model PRIVATE
  base
  gguf
  glog::glog
)
