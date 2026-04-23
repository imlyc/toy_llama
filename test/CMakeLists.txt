find_package(GTest REQUIRED)

add_executable(tokenizer_test
  tokenizer_test.cc
)

target_include_directories(tokenizer_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(tokenizer_test PRIVATE
  base
  gguf
  libtlm
  glog::glog
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tokenizer_test)
