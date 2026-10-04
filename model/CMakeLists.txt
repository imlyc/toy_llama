add_library(model OBJECT
  model.cc
  model_loader.cc
  token.cc
)

target_link_libraries(model PRIVATE
  absl::check
  absl::log
  base
  gguf
)
