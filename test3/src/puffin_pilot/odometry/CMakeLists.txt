cmake_minimum_required(VERSION 2.8.3)
project(odometry)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(-std=c++11)
set(CMAKE_BUILD_TYPE Release)

#############
# LIBRARIES #
#############
cs_add_library(imu_orientation_filter_madgwick_lib
  src/imu_orientation_filter_madgwick.cpp
)

cs_add_library(imu_orientation_filter_valenti_lib
  src/imu_orientation_filter_valenti.cpp
)

############
# BINARIES #
############
cs_add_executable(odometry_node
  src/odometry_node.cpp
)

target_link_libraries(odometry_node
  imu_orientation_filter_madgwick_lib
  imu_orientation_filter_valenti_lib
)

#add_dependencies(odometry_node ${PROJECT_NAME}_gencfg)

##########
# EXPORT #
##########
cs_install()
cs_export()
