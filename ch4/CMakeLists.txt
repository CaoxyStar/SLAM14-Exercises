cmake_minimum_required(VERSION 2.8)
project(useSophus)

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
find_package(Sophus  REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

add_executable(useSophus useSophus.cpp)
target_link_libraries(useSophus ${Pangolin_LIBRARIES})
target_link_libraries(useSophus ${Sophus_LIBRARIES} fmt)

add_executable(error error.cpp)
target_link_libraries(error ${Pangolin_LIBRARIES})
target_link_libraries(error ${Sophus_LIBRARIES} fmt)
