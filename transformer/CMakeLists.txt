add_library(transformer OBJECT
  decoder_block.cc
  group_query_attn_block.cc
  linear_layer.cc
  rms_norm_layer.cc
  swiglu_ffn_block.cc
  transformer.cc
)

target_link_libraries(transformer PRIVATE
  tensor
  glog::glog
)
