#include <ros/ros.h>
#include <iostream>
#include <nav_msgs/Odometry.h>
#include <mav_msgs/RateThrust.h>
#include <mav_msgs/RollPitchYawrateThrust.h>
#include <mav_msgs/eigen_mav_msgs.h>
#include <mav_msgs/conversions.h>

#include <dynamic_reconfigure/server.h>
#include <rate_thrust_controller/RateThrustControllerConfig.h>

using namespace std;


double p_gain_roll;
double p_gain_pitch;
double i_gain_roll;
double i_gain_pitch;
double d_gain_roll;
double d_gain_pitch;

bool got_first_rpyt_command = false;

mav_msgs::RollPitchYawrateThrust roll_pitch_yawthrust;
ros::Publisher rate_thrust_node;

void dyn_config_callback(rate_thrust_controller::RateThrustControllerConfig &config, uint32_t level)
{
    ROS_INFO("Set config: Roll PID(%.2f, %.2f, %.2f) Pitch PID(%.2f, %.2f, %.2f)",
                config.p_gain_roll,
                config.p_gain_pitch,
                config.i_gain_roll,
                config.i_gain_pitch,
                config.d_gain_roll,
                config.d_gain_pitch);

    p_gain_roll  = config.p_gain_roll;
    p_gain_pitch = config.p_gain_pitch;
    i_gain_roll  = config.i_gain_roll;
    i_gain_pitch = config.i_gain_pitch;
    d_gain_roll  = config.d_gain_roll;
    d_gain_pitch = config.d_gain_pitch;
}

void rollPitchYawrateThrustCallback(const mav_msgs::RollPitchYawrateThrust& msg)
{
    ROS_INFO_ONCE("RateThrustController got first roll-pitch-yawrate-thrust message.");

    roll_pitch_yawthrust = msg;

    got_first_rpyt_command = true;
}

void OdometryCallback(const nav_msgs::Odometry& msg)
{
    ROS_INFO_ONCE("RateThrustController got first odometry message.");

    if (!got_first_rpyt_command) {
        return;
    }

    mav_msgs::EigenOdometry odometry;
    eigenOdometryFromMsg(msg, &odometry);

    Eigen::Vector3d current_rpy;
    odometry.getEulerAngles(&current_rpy);

    // Proportional
    double p_error_roll  = roll_pitch_yawthrust.roll  - current_rpy(0);
    double p_error_pitch = roll_pitch_yawthrust.pitch - current_rpy(1);

    // Integral
    static double i_error_roll = 0;
    static double i_error_pitch = 0;
    i_error_roll  += p_error_roll;
    i_error_pitch += p_error_pitch;

    // Derivative
    static double p_error_roll_previous = 0;
    static double p_error_pitch_previous = 0;
    double d_error_roll  = p_error_roll  - p_error_roll_previous;
    double d_error_pitch = p_error_pitch - p_error_pitch_previous;

    // Calculate desired velocities
    double vel_roll, vel_pitch, vel_yaw;
    vel_roll  = p_error_roll  * p_gain_roll  + i_error_roll  * i_gain_roll  + d_error_roll  * d_gain_roll;
    vel_pitch = p_error_pitch * p_gain_pitch + i_error_pitch * i_gain_pitch + d_error_pitch * d_gain_pitch;
    vel_yaw = roll_pitch_yawthrust.yaw_rate;

    // Rate thrust command
    mav_msgs::RateThrust rate_thrust_msg;
    rate_thrust_msg.header.stamp = msg.header.stamp;
    rate_thrust_msg.angular_rates.x = vel_roll;
    rate_thrust_msg.angular_rates.y = vel_pitch;
    rate_thrust_msg.angular_rates.z = vel_yaw;
    rate_thrust_msg.thrust = roll_pitch_yawthrust.thrust;

    rate_thrust_node.publish(rate_thrust_msg);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "rate_thrust_controller");

    ros::NodeHandle subscriber_node;
    ros::Subscriber odometry_node = subscriber_node.subscribe("odometry",                  10, &OdometryCallback);
    ros::Subscriber rpyt_node     = subscriber_node.subscribe("roll_pitch_yawrate_thrust", 10, &rollPitchYawrateThrustCallback);

    ros::NodeHandle publisher_node;
    rate_thrust_node = publisher_node.advertise<mav_msgs::RateThrust>("rate_thrust", 1);

    dynamic_reconfigure::Server<rate_thrust_controller::RateThrustControllerConfig> controller_dyn_config_server_;
    controller_dyn_config_server_.setCallback(&dyn_config_callback);


        ros::spinOnce();
    // Rate thrust command
    mav_msgs::RateThrust rate_thrust_msg;
    rate_thrust_msg.angular_rates.x = 0;
    rate_thrust_msg.angular_rates.y = 0;
    rate_thrust_msg.angular_rates.z = 0;
    rate_thrust_msg.thrust.x = 0;
    rate_thrust_msg.thrust.y = 0;
    rate_thrust_msg.thrust.z = 0;


    int count = 0;

    ros::Rate loop_rate(1);
    while (ros::ok()) {

        if (count < 3) {
            count += 1;
            rate_thrust_msg.header.stamp = ros::Time::now();
            rate_thrust_node.publish(rate_thrust_msg);
            rate_thrust_msg.thrust.z = 10;
        }

        ros::spinOnce();
        loop_rate.sleep();
    }
}
