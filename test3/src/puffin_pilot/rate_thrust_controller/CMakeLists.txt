cmake_minimum_required(VERSION 2.8.3)
project(rate_thrust_controller)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

#############
# LIBRARIES #
#############


############
# BINARIES #
############
cs_add_executable(rate_thrust_controller_node
  src/rate_thrust_controller_node.cpp
)

add_dependencies(rate_thrust_controller_node ${${PROJECT_NAME}_EXPORTED_TARGETS})

##########
# EXPORT #
##########
cs_install()
cs_export()
