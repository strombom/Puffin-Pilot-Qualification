#include <ros/ros.h>
#include <iostream>
#include <nav_msgs/Odometry.h>
#include <mav_msgs/RateThrust.h>


using namespace std;

ros::Time last_odometry_callback;


void OdometryCallback(const nav_msgs::Odometry& msg)
{
    ROS_INFO_ONCE("PuffinLauncherNode got first odometry message.");

    last_odometry_callback = msg.header.stamp;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "puffin_launcher");

    ros::NodeHandle subscriber_node;
    ros::Subscriber odometry_node = subscriber_node.subscribe("odometry", 0, &OdometryCallback);

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

    last_odometry_callback = ros::Time::now() + ros::Duration(6.0);

    ros::Rate loop_rate(10);
    while (ros::ok()) {

        double delta_time = (ros::Time::now() - last_odometry_callback).toSec();
        if (delta_time > 1.0) {
            ROS_INFO("RateThrustController launching.");
            last_odometry_callback = ros::Time::now();
            rate_thrust_msg.angular_rates.y = 1;
            rate_thrust_msg.thrust.z = 11;
            rate_thrust_msg.header.stamp = ros::Time::now();
            rate_thrust_node.publish(rate_thrust_msg);
        }

        ros::spinOnce();
        loop_rate.sleep();
    }
}
