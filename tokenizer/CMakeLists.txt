add_library(tokenizer OBJECT
  chat_template.cc
  tokenizer.cc
)

target_link_libraries(tokenizer PRIVATE
  glog::glog
  re2::re2
)
