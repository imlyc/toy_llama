add_library(sampler OBJECT
  sampler.cc
)

target_link_libraries(sampler PRIVATE
  glog::glog
)
