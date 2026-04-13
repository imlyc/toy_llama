add_library(gguf OBJECT
  gguf_parser.cc
)

target_link_libraries(gguf PRIVATE
  glog::glog
)
