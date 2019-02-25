cmake_minimum_required(VERSION 2.8.3)
project(pace_notes)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

set(CMAKE_MACOSX_RPATH 0)
add_definitions(-std=c++11)

############
# BINARIES #
############
cs_add_executable(pace_notes_node
  src/pace_notes_node.cpp
)

#target_link_libraries(pace_notes_node ${PROJECT_NAME})

##########
# EXPORT #
##########
cs_install()
cs_export()
