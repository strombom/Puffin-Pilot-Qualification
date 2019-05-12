cmake_minimum_required(VERSION 2.8.3)
project(trajectory_generator)

find_package(catkin REQUIRED)


add_definitions(-std=c++11)
set(CMAKE_BUILD_TYPE Release)

catkin_package(
   INCLUDE_DIRS 
   LIBRARIES ${PROJECT_NAME}
   CATKIN_DEPENDS roscpp)



#############
# LIBRARIES #
#############

############
# BINARIES #
############
set_target_properties(trajectory_generator_node
                      PROPERTIES 
                      PREFIX "")

##########
# EXPORT #
##########

