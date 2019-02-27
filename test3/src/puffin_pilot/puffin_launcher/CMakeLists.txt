cmake_minimum_required(VERSION 2.8.3)
project(puffin_launcher)

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
cs_add_executable(puffin_launcher_node
  src/puffin_launcher_node.cpp
)

#add_dependencies(puffin_launcher_node ${${PROJECT_NAME}_EXPORTED_TARGETS})

##########
# EXPORT #
##########
cs_install()
cs_export()
