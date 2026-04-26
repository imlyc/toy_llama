add_library(tokenizer OBJECT
  tokenizer.cc
)

target_link_libraries(tokenizer PRIVATE
  glog::glog
  re2::re2
)
