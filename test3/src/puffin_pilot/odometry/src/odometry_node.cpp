#include <ros/ros.h>
#include <Eigen/Geometry>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include "imu_orientation_filter_madgwick.h"
#include "imu_orientation_filter_valenti.h"

using namespace std;
using namespace imu_orientation_filter_valenti_ns;


ImuOrientationFilterValenti  imu_orientation_filter_valenti;
ImuOrientationFilterMadgwick imu_orientation_filter_madgwick;

struct StateEstimate {
    ros::Time       timestamp;
    tf2::Vector3    position;
    tf2::Vector3    velocity;
    bool            valid;
};

StateEstimate imu_state;
StateEstimate ir_odometer_state;

static bool imu_filter_initialized = false;
static bool ir_odometer_initialized = false;

const static tf2::Vector3 gravity(0.0, 0.0, 9.81);
const static bool use_madgwick = true;

std::vector<double> initial_pose;

ros::Publisher ir_ok_node;
ros::Publisher pitch_node;
ros::Publisher puffin_odometry_node;
ros::Publisher puffin_acc_node;
//ros::Publisher puffin_odometry_gt_out_node;

bool measure_ir = false;

tf2::Quaternion look_at_quaternion(tf2::Vector3 direction)
{
    Eigen::Vector3d to_vector(direction.x(), direction.y(), direction.z());
    to_vector = to_vector.normalized();
    Eigen::Vector3d rot_axis = Eigen::Vector3d(1,0,0).cross(to_vector);
    double dot = Eigen::Vector3d(1,0,0).dot(to_vector);
    return tf2::Quaternion(rot_axis.x(), rot_axis.y(), rot_axis.z(), dot+1);
}

/*
void odometry_gt_callback(const nav_msgs::OdometryConstPtr& odom_gt_msg) {
    ROS_INFO_ONCE("Odometry got first OdometryGT message.");
    
    tf2::Vector3    position    = tf2::Vector3(odom_gt_msg->pose.pose.position.x, odom_gt_msg->pose.pose.position.y, odom_gt_msg->pose.pose.position.z);
    tf2::Vector3    linear      = tf2::Vector3(odom_gt_msg->twist.twist.linear.x, odom_gt_msg->twist.twist.linear.y, odom_gt_msg->twist.twist.linear.z);
    tf2::Vector3    angular     = tf2::Vector3(odom_gt_msg->twist.twist.angular.x, odom_gt_msg->twist.twist.angular.y, odom_gt_msg->twist.twist.angular.z);
    tf2::Quaternion orientation = tf2::Quaternion(odom_gt_msg->pose.pose.orientation.x, odom_gt_msg->pose.pose.orientation.y, odom_gt_msg->pose.pose.orientation.z, odom_gt_msg->pose.pose.orientation.w);

    tf2::Transform t_rot(orientation.inverse());
    linear  = t_rot * linear;
    angular = t_rot * angular;

    nav_msgs::Odometry odom;
    odom.header.stamp = odom_gt_msg->header.stamp;
    odom.header.frame_id = "puffin_nest";
    odom.child_frame_id = "odom_gt";

    odom.pose.pose.position.x    = position.x();
    odom.pose.pose.position.y    = position.y();
    odom.pose.pose.position.z    = position.z();
    odom.pose.pose.orientation.w = orientation.w();
    odom.pose.pose.orientation.x = orientation.x();
    odom.pose.pose.orientation.y = orientation.y();
    odom.pose.pose.orientation.z = orientation.z();
    odom.twist.twist.linear.x    = linear.x();
    odom.twist.twist.linear.y    = linear.y();
    odom.twist.twist.linear.z    = linear.z();
    odom.twist.twist.angular.x   = angular.x();
    odom.twist.twist.angular.y   = angular.y();
    odom.twist.twist.angular.z   = angular.z();

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            odom.pose.covariance[i*6 + j] = 0;
            odom.twist.covariance[i*6 + j] = 0;
        }
        odom.pose.covariance[i*6 + i] = 0;
        odom.twist.covariance[i*6 + i] = 0;
    }

    puffin_odometry_gt_out_node.publish(odom);
}
*/

void uav_imu_callback(const sensor_msgs::ImuConstPtr &msg) {
    ROS_INFO_ONCE("Odometry got first IMU message.");

    if (!imu_filter_initialized) {
        // First timestamp.
        imu_state.timestamp = msg->header.stamp;

        // Set initial state.
        tf2::Quaternion initial_orientation = tf2::Quaternion(initial_pose.at(3), initial_pose.at(4), initial_pose.at(5), initial_pose.at(6));
        imu_state.position                  = tf2::Vector3(initial_pose.at(0), initial_pose.at(1), initial_pose.at(2));
        imu_state.velocity                  = tf2::Vector3(0.0, 0.0, 0.0);

        // Initialize orientation filter.
        if (use_madgwick) {
            imu_orientation_filter_madgwick.setOrientation(initial_orientation.w(), 
                                                           initial_orientation.x(), 
                                                           initial_orientation.y(), 
                                                           initial_orientation.z());
        } else {
            imu_orientation_filter_valenti.setOrientation (initial_orientation.w(), 
                                                           initial_orientation.x(), 
                                                           initial_orientation.y(), 
                                                           initial_orientation.z());
        }
        imu_filter_initialized = true;
        return;
    }

    double delta_time = (msg->header.stamp - imu_state.timestamp).toSec();
    if (delta_time <= 0) {
        // IMU callback arrived in wrong order.
        return;
    }

    // Update orientation.
    double qx, qy, qz, qw;
    if (use_madgwick) {
        imu_orientation_filter_madgwick.madgwickAHRSupdateIMU(msg->angular_velocity.x,    msg->angular_velocity.y,    msg->angular_velocity.z,
                                                              msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z,
                                                              delta_time);
        imu_orientation_filter_madgwick.getOrientation(qw, qx, qy, qz);
    } else {
        imu_orientation_filter_valenti.update(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z,
                                              msg->angular_velocity.x,    msg->angular_velocity.y,    msg->angular_velocity.z,
                                              delta_time);
        imu_orientation_filter_valenti.getOrientation(qw, qx, qy, qz);
    }

    tf2::Quaternion orientation  = tf2::Quaternion(qx, qy, qz, qw);
    
    tf2::Vector3 acceleration = tf2::Transform(orientation) * tf2::Vector3(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);

    // Update state.
    StateEstimate state_new;
    state_new.timestamp = msg->header.stamp;
    state_new.velocity  = imu_state.velocity + delta_time * (acceleration - gravity);
    state_new.position  = imu_state.position + delta_time * (state_new.velocity);

    // Save new state.
    imu_state = state_new;

    // Broadcast new state.
    geometry_msgs::TransformStamped tf_imu;
    tf_imu.header.stamp = msg->header.stamp;
    tf_imu.header.frame_id = "puffin_nest";
    tf_imu.child_frame_id  = "puffin_imu";
    tf_imu.transform.translation.x = imu_state.position.x();
    tf_imu.transform.translation.y = imu_state.position.y();
    tf_imu.transform.translation.z = imu_state.position.z();
    tf_imu.transform.rotation.w    = orientation.w();
    tf_imu.transform.rotation.x    = orientation.x();
    tf_imu.transform.rotation.y    = orientation.y();
    tf_imu.transform.rotation.z    = orientation.z();
    static tf2_ros::TransformBroadcaster tf_broadcaster;
    tf_broadcaster.sendTransform(tf_imu);

    nav_msgs::Odometry odom;
    odom.header.stamp = msg->header.stamp;
    odom.header.frame_id = "puffin_nest";
    odom.child_frame_id = "odom";

    tf2::Transform t_rot(orientation.inverse());
    tf2::Vector3 velocity = t_rot * imu_state.velocity;

    geometry_msgs::Vector3 acc_msg;
    acc_msg.x = (acceleration - 1*gravity).x();
    acc_msg.y = (acceleration - 1*gravity).y();
    acc_msg.z = (acceleration - 1*gravity).z();
    puffin_acc_node.publish(acc_msg);

    odom.pose.pose.position.x    = imu_state.position.x();
    odom.pose.pose.position.y    = imu_state.position.y();
    odom.pose.pose.position.z    = imu_state.position.z();
    odom.pose.pose.orientation.w = orientation.w();
    odom.pose.pose.orientation.x = orientation.x();
    odom.pose.pose.orientation.y = orientation.y();
    odom.pose.pose.orientation.z = orientation.z();
    odom.twist.twist.linear.x    = velocity.x();
    odom.twist.twist.linear.y    = velocity.y();
    odom.twist.twist.linear.z    = velocity.z();
    odom.twist.twist.angular.x   = msg->angular_velocity.x;
    odom.twist.twist.angular.y   = msg->angular_velocity.y;
    odom.twist.twist.angular.z   = msg->angular_velocity.z;

    puffin_odometry_node.publish(odom);
    ROS_INFO_ONCE("Odometry sent first odometry message.");

    std_msgs::Float32 pitch_msg;
    pitch_msg.data = asin(-2.0*(orientation.x() * orientation.z() - orientation.w() * orientation.y()));
    pitch_node.publish(pitch_msg);
}

int measure_ir_count = -1;

void ir_trig_callback(const std_msgs::Bool trig)
{
    //ROS_INFO("Odom: Measure IR markers start.");
    measure_ir_count = 0;
}

void ir_marker_odometry_callback(const geometry_msgs::PoseStamped& msg)
{
    ROS_INFO_ONCE("Odometry got first IR marker pose message.");

    if (!imu_filter_initialized) {
        return;
    }

    tf2::Vector3 new_position(msg.pose.position.x,
                              msg.pose.position.y,
                              msg.pose.position.z);

    static tf2::Vector3 previous_position;

    if (!ir_odometer_initialized) {
        ir_odometer_state.timestamp = msg.header.stamp;
        ir_odometer_state.position  = new_position;
        ir_odometer_state.velocity  = tf2::Vector3(0, 0, 0);
        ir_odometer_state.valid     = false;
        ir_odometer_initialized     = true;
        previous_position = new_position;
        return;
    }

    if (measure_ir_count == -2) {
        return;
    }
    if (measure_ir_count >= 0) {
        measure_ir_count++;
    }

    double delta_time = (msg.header.stamp - ir_odometer_state.timestamp).toSec();
    if (delta_time <= 0) {
        // Pose callback out of order.
        return;
    }

    static const float max_ir_odom_interval = 0.04;
    if (delta_time > max_ir_odom_interval) {
        // Too far between pose callbacks.
        ir_odometer_state.valid    = false;
    } else {
        ir_odometer_state.velocity = (new_position - ir_odometer_state.position) / delta_time;
        ir_odometer_state.valid    = true;

        imu_state.velocity = (ir_odometer_state.velocity + 31 * imu_state.velocity) / 32;
        imu_state.position = (ir_odometer_state.position + 15 * imu_state.position) / 16;

        if (measure_ir_count > 15) {
            ir_ok_node.publish(true);
            measure_ir_count = -1;
        }
    }

    ir_odometer_state.timestamp = msg.header.stamp;
    ir_odometer_state.position  = new_position;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "puffin_state_estimation");

    ros::param::get("/uav/flightgoggles_uav_dynamics/init_pose", initial_pose);

    // Orientation filter setup.
    if (use_madgwick) {
        imu_orientation_filter_madgwick.setAlgorithmGain(0.0);
        imu_orientation_filter_madgwick.setWorldFrame(WorldFrame::NWU);
    } else {
        imu_orientation_filter_valenti.setGainAcc(0.1);
        imu_orientation_filter_valenti.setDoBiasEstimation(false);
        imu_orientation_filter_valenti.setDoAdaptiveGain(false);
    }

    ros::NodeHandle subscriber_node;
    ros::Subscriber imu_callback_node         = subscriber_node.subscribe("imu",             1, &uav_imu_callback,            ros::TransportHints().tcpNoDelay());
    ros::Subscriber irodom_callback_node      = subscriber_node.subscribe("ir_markers_pose", 1, &ir_marker_odometry_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber ir_trig_callback_node     = subscriber_node.subscribe("ir_trig",         1, &ir_trig_callback,            ros::TransportHints().tcpNoDelay());
    //ros::Subscriber odometry_gt_callback_node = subscriber_node.subscribe("odometry_gt_in",  1, &odometry_gt_callback,        ros::TransportHints().tcpNoDelay());
   
    ros::NodeHandle publisher_node;
    ir_ok_node                  = publisher_node.advertise<std_msgs::Bool>("ir_ok", 1, false);
    puffin_odometry_node        = publisher_node.advertise<nav_msgs::Odometry>("odometry", 1);
    puffin_acc_node             = publisher_node.advertise<geometry_msgs::Vector3>("acc", 1);
    pitch_node                  = publisher_node.advertise<std_msgs::Float32>("pitch", 1, false);
    //puffin_odometry_gt_out_node = publisher_node.advertise<nav_msgs::Odometry>("odometry_gt_out", 1);

    // REMOVE!!!    
    geometry_msgs::TransformStamped tf_nest;
    tf_nest.header.stamp = ros::Time::now();
    tf_nest.header.frame_id = "puffin_nest";
    tf_nest.child_frame_id = "world";
    tf_nest.transform.translation.x = 0;
    tf_nest.transform.translation.y = 0;
    tf_nest.transform.translation.z = 0;
    tf_nest.transform.rotation.x = 0;
    tf_nest.transform.rotation.y = 0;
    tf_nest.transform.rotation.z = 0;
    tf_nest.transform.rotation.w = 1;
    static tf2_ros::StaticTransformBroadcaster tf_static_broadcaster;
    tf_static_broadcaster.sendTransform(tf_nest);
    
    ros::spin();
    return 0;
}
