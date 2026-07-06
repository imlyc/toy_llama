find_package(GTest REQUIRED)

add_executable(toy_llama_test
  tokenizer_test.cc
  gguf_parser_test.cc
)

target_include_directories(toy_llama_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(toy_llama_test PRIVATE
  base
  gguf
  model
  tensor
  tokenizer
  glog::glog
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(toy_llama_test)
