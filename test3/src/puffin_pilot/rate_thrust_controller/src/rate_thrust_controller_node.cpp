#include <ros/ros.h>
#include <iostream>
#include <nav_msgs/Odometry.h>
#include <mav_msgs/RateThrust.h>
#include <mav_msgs/RollPitchYawrateThrust.h>
#include <mav_msgs/eigen_mav_msgs.h>
#include <mav_msgs/conversions.h>

#include <dynamic_reconfigure/server.h>
#include <rate_thrust_controller/RateThrustControllerConfig.h>

#include <geometry_msgs/TwistStamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Matrix3x3.h>

using namespace std;


double p_gain_roll;
double p_gain_pitch;
double i_gain_roll;
double i_gain_pitch;
double d_gain_roll;
double d_gain_pitch;

bool got_first_rpyt_command = false;

//mav_msgs::RollPitchYawrateThrust roll_pitch_yawthrust;
mav_msgs::RateThrust rate_thrust;
ros::Publisher rate_thrust_node;
ros::Publisher rtc_pose_pub;

ros::Time last_odometry_callback;


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

void rollPitchYawrateThrustCallback(const mav_msgs::RateThrust& msg)
{
    ROS_INFO_ONCE("RateThrustController got first roll-pitch-yawrate-thrust message.");

    rate_thrust = msg;

    got_first_rpyt_command = true;
}
/*
void rollPitchYawrateThrustCallback(const mav_msgs::RollPitchYawrateThrust& msg)
{
    ROS_INFO_ONCE("RateThrustController got first roll-pitch-yawrate-thrust message.");

    roll_pitch_yawthrust = msg;

    got_first_rpyt_command = true;
}
*/

void OdometryCallback(const nav_msgs::Odometry& msg)
{
    ROS_INFO_ONCE("RateThrustController got first odometry message.");

    if (!got_first_rpyt_command) {
        return;
    }

    last_odometry_callback = msg.header.stamp;

    static mav_msgs::EigenOdometry odometry;
    eigenOdometryFromMsg(msg, &odometry);

    static Eigen::Vector3d current_rpy;
    odometry.getEulerAngles(&current_rpy);

    // Proportional
    //double p_error_roll  = +(roll_pitch_yawthrust.roll  - current_rpy(0));
    //double p_error_pitch = +(roll_pitch_yawthrust.pitch - current_rpy(1));
    double error_roll  = rate_thrust.angular_rates.x - current_rpy(0);
    double error_pitch = rate_thrust.angular_rates.y - current_rpy(1);

    /*
    if (p_error_roll > 0)
        p_error_roll = pow(p_error_roll, 2.0);
    if (p_error_roll < 0)
        p_error_roll = -pow(p_error_roll, 2.0);
    */

    // Integral
    static double i_error_roll = 0;
    static double i_error_pitch = 0;
    i_error_roll  += error_roll;
    i_error_pitch += error_pitch;

    // Anti wind-up
    double max_i_error_roll  = 2.0 / i_gain_roll;
    double max_i_error_pitch = 2.0 / i_gain_pitch;
    if (abs(i_error_roll) > max_i_error_roll) {
        i_error_roll = max_i_error_roll * i_error_roll / abs(i_error_roll);
    }
    if (abs(i_error_pitch) > max_i_error_pitch) {
        i_error_pitch = max_i_error_pitch * i_error_pitch / abs(i_error_pitch);
    }

    double i_term_roll  = i_gain_roll  * i_error_roll;
    double i_term_pitch = i_gain_pitch * i_error_pitch;

    // Derivative
    static double add = 0.95;
    static double error_roll_avg_previous = error_roll;
    static double error_pitch_avg_previous = error_pitch;
    //static double d_error_roll_previous = 0;
    //static double d_error_pitch_previous = 0;
    static double error_roll_avg  = error_roll;
    static double error_pitch_avg = error_pitch;
    error_roll_avg =  error_roll  * add + (1 - add) * error_roll_avg;
    error_pitch_avg = error_pitch * add + (1 - add) * error_pitch_avg;

    double d_error_roll  = error_roll_avg - error_roll_avg_previous;  // d_error_roll_previous - error_roll;
    double d_error_pitch = error_pitch_avg - error_pitch_avg_previous; // d_error_pitch_previous - error_pitch;
    error_roll_avg_previous  = error_roll_avg;
    error_pitch_avg_previous = error_pitch_avg;

    // Derivative ceiling
    double max_d_error_roll  = 3.0 / d_gain_roll;
    double max_d_error_pitch = 3.0 / d_gain_pitch;
    if (abs(d_error_roll) > max_d_error_roll) {
        d_error_roll = max_d_error_roll * d_error_roll / abs(d_error_roll);
    }
    if (abs(d_error_pitch) > max_d_error_pitch) {
        d_error_pitch = max_d_error_pitch * d_error_pitch / abs(d_error_pitch);
    }

    //xdk_roll  = add * xdk_roll  + (1 - add) * current_rpy(0);
    //xdk_pitch = add * xdk_pitch + (1 - add) * current_rpy(1);
    //d_error_roll_previous  += d_error_roll;
    //d_error_pitch_previous += d_error_pitch;

    // Calculate desired velocities
    double vel_roll, vel_pitch, vel_yaw;
    vel_roll  = error_roll  * p_gain_roll  + i_error_roll  * i_gain_roll  + d_error_roll  * d_gain_roll;
    vel_pitch = error_pitch * p_gain_pitch + i_error_pitch * i_gain_pitch + d_error_pitch * d_gain_pitch;
    vel_yaw = rate_thrust.angular_rates.z;


    /*
    double vel_roll, vel_pitch, vel_yaw;
    vel_roll = roll_pitch_yawthrust.roll;
    vel_pitch = roll_pitch_yawthrust.pitch;
    vel_yaw = roll_pitch_yawthrust.yaw_rate;
    */

    // Rate thrust command
    static mav_msgs::RateThrust rate_thrust_msg;
    rate_thrust_msg.header.stamp = msg.header.stamp;
    rate_thrust_msg.angular_rates.x = vel_roll;
    rate_thrust_msg.angular_rates.y = vel_pitch;
    rate_thrust_msg.angular_rates.z = vel_yaw;
    rate_thrust_msg.thrust = rate_thrust.thrust;

    rate_thrust_node.publish(rate_thrust_msg);

/*
    static int counter = 0;
    if (counter > 100) {
        ROS_INFO_STREAM(
            "RTC: R(" << vel_roll << ") (" << vel_pitch << ") (" << vel_yaw << ") "
                    << " z(" << rate_thrust.thrust.z << ") ");
      counter = 0;
    }
    counter++;
*/


/*
    static tf2_ros::Buffer tfBuffer;
    static tf2_ros::TransformListener rtc_pose_transform_listener(tfBuffer);
    
    geometry_msgs::TransformStamped transformStamped;

    try{
        transformStamped = tfBuffer.lookupTransform("world", "uav/imu", ros::Time(0));

        tf2::Quaternion q(
            transformStamped.transform.rotation.x,
            transformStamped.transform.rotation.y,
            transformStamped.transform.rotation.z,
            transformStamped.transform.rotation.w);
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        geometry_msgs::TwistStamped ts;
        ts.header.stamp = msg.header.stamp;
        ts.twist.angular.x = roll;
        ts.twist.angular.y = pitch;
        ts.twist.angular.z = yaw;
        rtc_pose_pub.publish(ts);
    }
    catch(tf2::TransformException &ex) {

    }
*/
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "rate_thrust_controller");

    ros::NodeHandle subscriber_node;
    ros::Subscriber odometry_node = subscriber_node.subscribe("odometry",                  50, &OdometryCallback);
    //ros::Subscriber rpyt_node     = subscriber_node.subscribe("roll_pitch_yawrate_thrust", 10, &rollPitchYawrateThrustCallback);
    ros::Subscriber rpyt_node     = subscriber_node.subscribe("roll_pitch_yawrate_thrust", 50, &rollPitchYawrateThrustCallback);

    ros::NodeHandle publisher_node;
    rate_thrust_node = publisher_node.advertise<mav_msgs::RateThrust>("rate_thrust", 500);
    rtc_pose_pub     = publisher_node.advertise<geometry_msgs::TwistStamped>("rtc_pose", 500);

    dynamic_reconfigure::Server<rate_thrust_controller::RateThrustControllerConfig> controller_dyn_config_server_;
    controller_dyn_config_server_.setCallback(&dyn_config_callback);


    // Rate thrust command
    mav_msgs::RateThrust rate_thrust_msg;
    rate_thrust_msg.angular_rates.x = 0;
    rate_thrust_msg.angular_rates.y = 0;
    rate_thrust_msg.angular_rates.z = 0;
    rate_thrust_msg.thrust.x = 0;
    rate_thrust_msg.thrust.y = 0;
    rate_thrust_msg.thrust.z = 0;

    last_odometry_callback = ros::Time::now();

    ros::spin();


    ros::Rate loop_rate(1000);
    while (ros::ok()) {

        double delta_time = (ros::Time::now() - last_odometry_callback).toSec();
        if (delta_time > 1.0) {
            ROS_INFO("RateThrustController launching.");
            last_odometry_callback = ros::Time::now();
            rate_thrust_msg.thrust.z = 10;
            rate_thrust_msg.header.stamp = ros::Time::now();
            rate_thrust_node.publish(rate_thrust_msg);
        }

        ros::spinOnce();
        loop_rate.sleep();
    }
}
