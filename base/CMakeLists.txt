add_library(base OBJECT
  mmap_file.cc
  scoped_fd.cc
)

target_link_libraries(base PRIVATE
  glog::glog
)
