cmake_minimum_required(VERSION 2.8.3)
project(puffin_ir_marker_odometry)

set(CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS roscpp geometry_msgs sensor_msgs tf2 tf2_ros geometry_msgs cv_bridge puffin_pace_notes)
find_package(SDL REQUIRED)
set(LIBS ${SDL_LIBRARY})

#######################################
## Declare ROS messages and services ##
#######################################

#add_message_files(
#  FILES
#  MyMessage.msg
#)

#generate_messages(
#  DEPENDENCIES
#  geometry_msgs
#)

###################################
## catkin specific configuration ##
###################################

find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED core calib3d)

catkin_package(
  #INCLUDE_DIRS include
  #LIBRARIES mylibrary
  CATKIN_DEPENDS roscpp geometry_msgs sensor_msgs tf2 tf2_ros cv_bridge
  DEPENDS ${LIBS}
)

###########
## Build ##
###########

include_directories(${catkin_INCLUDE_DIRS} ${SDL_INCLUDE_DIR})

add_executable(puffin_ir_marker_odometry src/main.cpp src/ir_marker_odometry.cpp src/rate_limiter.cpp)

target_link_libraries(puffin_ir_marker_odometry
  ${LIBS}
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  opencv_calib3d
)

add_dependencies(puffin_ir_marker_odometry ir_marker_odometry_gencpp rate_limiter_gencpp)

#############
## Install ##
#############


install(TARGETS puffin_ir_marker_odometry
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
