add_library(compute OBJECT
  compute_engine.cc
  storage.cc
)

target_link_libraries(compute PRIVATE
  tensor
  glog::glog
)
