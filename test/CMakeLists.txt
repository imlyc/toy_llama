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

# End-to-end golden + performance test against a real GGUF model. Kept out of
# the unit test binary and out of ctest: it needs a model file and an optimized
# build. Run it explicitly, e.g.
#   TOY_LLAMA_MODEL=data/models/llama-3.2-1b-instruct-q8_0.gguf \
#   TOY_LLAMA_GOLDEN=/tmp/toy_llama/e2e_golden_1b.txt \
#   ./build-release/test/toy_llama_e2e_test
add_executable(toy_llama_e2e_test
  e2e_test.cc
)

target_include_directories(toy_llama_e2e_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(toy_llama_e2e_test PRIVATE
  base
  compute
  gguf
  model
  sampler
  tensor
  tokenizer
  transformer
  libtlm
  glog::glog
  GTest::gtest_main
)
