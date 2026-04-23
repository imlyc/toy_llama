find_package(GTest REQUIRED)

add_executable(tokenizer_test
  tokenizer_test.cc
  ../tokenizer.cc
  ../model.cc
)

target_include_directories(tokenizer_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(tokenizer_test PRIVATE
  base
  gguf
  glog::glog
  re2::re2
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tokenizer_test)
