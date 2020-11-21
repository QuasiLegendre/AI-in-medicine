cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(FashionCNN)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(FashionCNN FashionCNN.cpp)
target_link_libraries(FashionCNN "${TORCH_LIBRARIES}")
set_property(TARGET FashionCNN PROPERTY CXX_STANDARD 14)
