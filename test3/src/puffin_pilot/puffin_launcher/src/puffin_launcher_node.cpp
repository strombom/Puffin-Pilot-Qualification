#include <ros/ros.h>
#include <iostream>
#include <nav_msgs/Odometry.h>
#include <mav_msgs/RateThrust.h>


using namespace std;


int main(int argc, char** argv)
{
    ros::init(argc, argv, "puffin_launcher");

    ros::NodeHandle publisher_node;
    ros::Publisher rate_thrust_node = publisher_node.advertise<mav_msgs::RateThrust>("rate_thrust", 10);

    // Rate thrust command
    mav_msgs::RateThrust rate_thrust_msg;
    rate_thrust_msg.angular_rates.x = 0;
    rate_thrust_msg.angular_rates.y = 0;
    rate_thrust_msg.angular_rates.z = 0;
    rate_thrust_msg.thrust.x = 0;
    rate_thrust_msg.thrust.y = 0;
    rate_thrust_msg.thrust.z = 0;

    ros::Rate loop_rate(1);
    loop_rate.sleep();
    loop_rate.sleep();
    /*
    loop_rate.sleep();
    loop_rate.sleep();
    loop_rate.sleep();
    loop_rate.sleep();
    loop_rate.sleep();
    loop_rate.sleep();
    loop_rate.sleep();
    */

    ROS_INFO("RateThrustController launching.");
    rate_thrust_msg.angular_rates.y = 1;
    rate_thrust_msg.thrust.z = 11;
    rate_thrust_msg.header.stamp = ros::Time::now();
    rate_thrust_node.publish(rate_thrust_msg);
    ros::spinOnce();

    ros::waitForShutdown();
}
