add_library(tokenizer OBJECT
  chat_template.cc
  tokenizer.cc
)

target_link_libraries(tokenizer PRIVATE
  absl::check
  absl::log
  re2::re2
)
