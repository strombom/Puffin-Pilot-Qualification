cmake_minimum_required(VERSION 2.8.3)
project(trajectory_generator)

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
cs_add_executable(trajectory_generator_node
  src/trajectory_generator_node.cpp
)

##########
# EXPORT #
##########
cs_install()
cs_export()
