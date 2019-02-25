cmake_minimum_required(VERSION 2.8.3)
project(ir_marker_localizer)

find_package(catkin_simple REQUIRED)
find_package(OpenCV REQUIRED core calib3d)
catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(-std=c++11)
set(CMAKE_BUILD_TYPE Release)

#############
# LIBRARIES #
#############
cs_add_library(rate_limiter_lib
  src/rate_limiter.cpp
)

############
# BINARIES #
############
cs_add_executable(ir_marker_localizer_node
  src/ir_marker_localizer_node.cpp
)

target_link_libraries(ir_marker_localizer_node
	rate_limiter_lib
	opencv_calib3d
)

##########
# EXPORT #
##########
cs_install()
cs_export()
