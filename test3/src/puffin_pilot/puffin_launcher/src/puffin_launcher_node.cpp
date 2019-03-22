#include <ros/ros.h>
#include <iostream>
#include <mav_msgs/RateThrust.h>
#include <flightgoggles/IRMarkerArray.h>


using namespace std;

bool received_ir_beacons = false;

void ir_beacons_callback(const flightgoggles::IRMarkerArrayConstPtr& ir_beacons_array)
{
    received_ir_beacons = true;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "puffin_launcher");

    ros::NodeHandle nh;
    ros::Publisher rate_thrust_node        = nh.advertise<mav_msgs::RateThrust>("rate_thrust", 10);
    ros::Subscriber ir_beacons_subscriber  = nh.subscribe("ir_beacons",  10, &ir_beacons_callback);

    ros::Rate loop_rate(1);
    while (!received_ir_beacons) {
        loop_rate.sleep();
        ros::spinOnce();
    }

    loop_rate.sleep();
    loop_rate.sleep();
    loop_rate.sleep();
    loop_rate.sleep();
    loop_rate.sleep();
    loop_rate.sleep();
    
    // Rate thrust command
    mav_msgs::RateThrust rate_thrust_msg;
    rate_thrust_msg.angular_rates.x = 0;
    rate_thrust_msg.angular_rates.y = 0;
    rate_thrust_msg.angular_rates.z = 0;
    rate_thrust_msg.thrust.x = 0;
    rate_thrust_msg.thrust.y = 0;
    rate_thrust_msg.thrust.z = 0;

    ROS_INFO("Puffin pilot launcher launching.");
    rate_thrust_msg.angular_rates.y = 1;
    rate_thrust_msg.thrust.z = 13;
    rate_thrust_msg.header.stamp = ros::Time::now();
    rate_thrust_node.publish(rate_thrust_msg);
    ros::spinOnce();

    ros::waitForShutdown();
}
