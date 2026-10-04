add_library(base OBJECT
  mmap_file.cc
  scoped_fd.cc
)

target_link_libraries(base PRIVATE
  absl::check
  absl::log
)
