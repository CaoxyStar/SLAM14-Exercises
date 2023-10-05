cmake_minimum_required(VERSION 2.8)
project(hhh)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(use_opencv_orb use_opencv_orb.cpp)
target_link_libraries(use_opencv_orb ${OpenCV_LIBS})

