add_library(tensor OBJECT
  dtype.cc
)

target_link_libraries(tensor PRIVATE
  base
  glog::glog
)
