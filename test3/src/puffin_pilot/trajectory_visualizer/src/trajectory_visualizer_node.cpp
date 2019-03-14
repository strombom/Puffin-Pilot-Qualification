#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <iostream>


using namespace std;


int main(int argc, char** argv)
{
    ros::init(argc, argv, "trajectory_visualizer");
    ros::start();

    ros::spin();
}

