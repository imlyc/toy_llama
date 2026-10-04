cmake_minimum_required(VERSION 3.28)

project(ToyLlama VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Avoid name conflict between absl::base and base. Other components still work
# fine since we run the binary from build tree.
set(CMAKE_SKIP_INSTALL_RULES ON)

# Allow sub folder to use absolute path for include
include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(base)
add_subdirectory(compute)
add_subdirectory(gguf)
add_subdirectory(model)
add_subdirectory(tensor)
add_subdirectory(tokenizer)
add_subdirectory(transformer)
add_subdirectory(sampler)

# Abseil is needed by re2, we also use it for logging.
# Build Abseil with the project's C++20, not its own default
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(third_party/abseil-cpp EXCLUDE_FROM_ALL)

add_subdirectory(third_party/eigen EXCLUDE_FROM_ALL)
add_subdirectory(third_party/re2 EXCLUDE_FROM_ALL)

add_library(libtlm OBJECT
  chat_engine.cc
)

target_link_libraries(libtlm PRIVATE
  base
  compute
  model
  tensor
  tokenizer
  transformer
  sampler
)

add_executable(toy_llama
  main.cc
)

target_link_libraries(toy_llama PRIVATE
  absl::log
  absl::log_initialize
  base
  compute
  gguf
  model
  tensor
  tokenizer
  transformer
  sampler
  libtlm
)

enable_testing()
add_subdirectory(test)
