cmake_minimum_required(VERSION 2.8.3)
project(puffin_state_estimation)

set(CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS roscpp geometry_msgs sensor_msgs tf2 tf2_ros geometry_msgs)
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

# Find Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

catkin_package(
  #INCLUDE_DIRS include
  #LIBRARIES mylibrary
  CATKIN_DEPENDS roscpp geometry_msgs sensor_msgs tf2 tf2_ros
  DEPENDS ${LIBS}
)

###########
## Build ##
###########

include_directories(${catkin_INCLUDE_DIRS} ${SDL_INCLUDE_DIR})

add_executable(puffin_state_estimation src/main.cpp src/imu_orientation_filter_madgwick.cpp src/imu_orientation_filter_valenti.cpp)

target_link_libraries(puffin_state_estimation
  ${LIBS}
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
)

add_dependencies(puffin_state_estimation imu_orientation_filter_madgwick_gencpp imu_orientation_filter_valenti_gencpp)

#############
## Install ##
#############


install(TARGETS puffin_state_estimation
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
