cmake_minimum_required(VERSION 4.2.1)

project(ToyLlama VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wthread-safety -fexperimental-library")
add_definitions(-D_LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS)

# glog
find_package(glog REQUIRED)

# Allow sub folder to use absolute path for include
include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(base)
add_subdirectory(gguf)

add_subdirectory(third_party/re2 EXCLUDE_FROM_ALL)

add_library(libtlm OBJECT
  chat_engine.cc
  model.cc
  model_loader.cc
  sampler.cc
  tokenizer.cc
  transformer.cc
)

target_link_libraries(libtlm PRIVATE
  base
  gguf

  glog::glog
  re2::re2
)

add_executable(toy_llama
  main.cc
)

target_link_libraries(toy_llama PRIVATE
  base
  gguf
  libtlm

  glog::glog
)

enable_testing()
add_subdirectory(test)
