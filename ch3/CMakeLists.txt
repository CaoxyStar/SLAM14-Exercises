cmake_minimum_required(VERSION 2.8)
project(plot)


find_package(Threads REQUIRED)
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_DIRECTORIES})
add_executable(plot plotTrajectory.cpp)
target_link_libraries(plot ${Pangolin_LIBRARIES})

add_executable(plot_learn pangolin_learn.cpp)
target_link_libraries(plot_learn ${Pangolin_LIBRARIES} Threads::Threads)
