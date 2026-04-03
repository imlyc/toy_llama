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

add_executable(toy_llama
  main.cc

  chat_engine.cc
  model.cc
  model_loader.cc
)

target_link_libraries(toy_llama PRIVATE
  glog::glog
)
