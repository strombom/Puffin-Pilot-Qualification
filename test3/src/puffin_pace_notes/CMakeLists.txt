cmake_minimum_required(VERSION 2.8.3)
project(puffin_pace_notes)

find_package(catkin REQUIRED COMPONENTS roscpp std_msgs message_generation)
find_package(SDL REQUIRED)
set(LIBS ${SDL_LIBRARY})

#######################################
## Declare ROS messages and services ##
#######################################

add_message_files(
  FILES
  PaceNote.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

###################################
## catkin specific configuration ##
###################################

catkin_package(
  #INCLUDE_DIRS include
  #LIBRARIES mylibrary
  CATKIN_DEPENDS roscpp std_msgs message_runtime
  DEPENDS ${LIBS}
)

###########
## Build ##
###########

include_directories(${catkin_INCLUDE_DIRS} ${SDL_INCLUDE_DIR})

add_executable(puffin_pace_notes src/main.cpp src/pace_notes.cpp)

target_link_libraries(puffin_pace_notes
  ${LIBS}
  ${catkin_LIBRARIES}
)

add_dependencies(puffin_pace_notes puffin_pace_notes_gencpp)

#############
## Install ##
#############


install(TARGETS puffin_pace_notes
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
