cmake_minimum_required(VERSION 2.8)
project(GaussNewton)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_Package(CSparse REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Ceres_INCLUDE_DIRS})
include_directories(${CSPARSE_INCLUDE_DIR})

LIST(APPEND CMAKE_MODULE_PATH /home/xiaoyu/model/g2o/cmake_modules)
SET(G2O_ROOT /home/xiaoyu/model/g2o)
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

add_executable(gaussnewton gaussnewton.cpp)
target_link_libraries(gaussnewton ${OpenCV_LIBS})

add_executable(useCeres useCeres.cpp)
target_link_libraries(useCeres ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(useg2o useG2O.cpp)
target_link_libraries(useg2o ${OpenCV_LIBS} g2o_core g2o_stuff)

