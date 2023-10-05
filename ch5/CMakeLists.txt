cmake_minimum_required(VERSION 2.8)
project(camera_use)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv_test opencv_test.cpp)
target_link_libraries(opencv_test ${OpenCV_LIBS})

