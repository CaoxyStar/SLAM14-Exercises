cmake_minimum_required(VERSION 2.8)
project(hhh)

set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.a" )

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(feature_training feature_training.cpp)
target_link_libraries(feature_training ${OpenCV_LIBS})
target_link_libraries(feature_training ${DBoW3_LIBS})

add_executable(loop_closure loop_closure.cpp)
target_link_libraries(loop_closure ${OpenCV_LIBS})
target_link_libraries(loop_closure ${DBoW3_LIBS})
