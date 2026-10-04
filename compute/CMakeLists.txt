add_library(compute OBJECT
  compute_engine.cc
)

target_link_libraries(compute PRIVATE
  absl::check
  absl::log
  eigen
  tensor
)
