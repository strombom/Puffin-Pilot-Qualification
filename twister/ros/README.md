# Puffin twister

Puffin twister ROS

## Tutorials

[ROS URDF](http://gazebosim.org/tutorials/?tut=ros_urdf)

## Quick Start

Rviz:

    roslaunch pt_description pt_rviz.launch

Gazebo:

    roslaunch pt_gazebo pt_world.launch

ROS Control:

    roslaunch pt_control pt_control.launch

Example of Moving Joints:

    rostopic pub /puffin_twister/joint2_position_controller/command std_msgs/Float64 "data: -0.9"

