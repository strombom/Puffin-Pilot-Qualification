cmake_minimum_required(VERSION 2.8.3)
project(trajectory_visualizer)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(-std=c++11)
set(CMAKE_BUILD_TYPE Release)

#############
# LIBRARIES #
#############

############
# BINARIES #
############
cs_add_executable(trajectory_visualizer_node
  src/trajectory_visualizer_node.cpp
)

##########
# EXPORT #
##########
cs_install()
cs_export()
