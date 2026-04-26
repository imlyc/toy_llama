add_library(transformer OBJECT
  transformer.cc
)

target_link_libraries(transformer PRIVATE
  glog::glog
)
