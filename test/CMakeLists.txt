find_package(GTest REQUIRED)

add_executable(toy_llama_test
  tokenizer_test.cc
  gguf_parser_test.cc
  tensor_view_test.cc
  compute_engine_test.cc
  transformer_test.cc
)

target_include_directories(toy_llama_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(toy_llama_test PRIVATE
  base
  compute
  gguf
  model
  tensor
  tokenizer
  transformer
  glog::glog
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(toy_llama_test DISCOVERY_MODE PRE_TEST)
