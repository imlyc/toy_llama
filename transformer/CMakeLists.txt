add_library(transformer OBJECT
  decoder_block.cc
  linear_layer.cc
  rms_norm_layer.cc
  transformer.cc
)

target_link_libraries(transformer PRIVATE
  tensor
  glog::glog
)
