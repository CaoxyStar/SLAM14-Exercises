cmake_minimum_required(VERSION 2.8)
project(hhh)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_DIRECTORIES})
find_package(Sophus  REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})
find_package(Threads REQUIRED)


add_executable(use_LK use_LK.cpp)
target_link_libraries(use_LK ${OpenCV_LIBS})

add_executable(direct_method direct_method.cpp)
target_link_libraries(direct_method ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES} fmt)
target_link_libraries(direct_method ${Pangolin_LIBRARIES})
target_link_libraries(direct_method ${Sophus_LIBRARIES} fmt)

add_executable(learn_pthread learn_pthread.cpp)
target_link_libraries(learn_pthread ${CMAKE_THREAD_LIBS_INIT})

