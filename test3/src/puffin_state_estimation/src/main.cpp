#include <ros/ros.h>
#include <Eigen/Geometry>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>

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
};

StateEstimate imu_state;
StateEstimate ir_odometer_state;

static bool imu_filter_initialized = false;
static bool ir_odometer_initialized = false;

const static tf2::Vector3 gravity(0.0, 0.0, 9.81);
const static bool use_madgwick = true;

std::vector<double> initial_pose;


void uavIMUCallback(const sensor_msgs::ImuConstPtr &msg) {

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
    tf2::Vector3    acceleration = tf2::Transform(orientation) * tf2::Vector3(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);

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
    tf_imu.transform.rotation.w = orientation.w();
    tf_imu.transform.rotation.x = orientation.x();
    tf_imu.transform.rotation.y = orientation.y();
    tf_imu.transform.rotation.z = orientation.z();
    static tf2_ros::TransformBroadcaster tf_broadcaster;
    tf_broadcaster.sendTransform(tf_imu);
}

void irMarkerOdometryCallback(const geometry_msgs::PoseStamped& msg)
{
    static const float max_ir_odom_interval = 0.1;

    tf2::Vector3 new_position(msg.pose.position.x,
                              msg.pose.position.y,
                              msg.pose.position.z);

    if (!ir_odometer_initialized) {
        ir_odometer_state.timestamp = msg.header.stamp;
        ir_odometer_state.position  = new_position;
        ir_odometer_state.velocity  = tf2::Vector3(0, 0, 0);
        ir_odometer_initialized     = true;
        return;
    }

    double delta_time = (msg.header.stamp - ir_odometer_state.timestamp).toSec();
    if (delta_time <= 0) {
        // Pose callback out of order.
        return;
    }
    if (delta_time > max_ir_odom_interval) {
        // Too far between pose callbacks.
        ir_odometer_state.timestamp = msg.header.stamp;
        ir_odometer_state.position  = new_position;
        return;
    }

    // Update state.
    imu_state.velocity = (new_position - ir_odometer_state.position) / delta_time;
    imu_state.position =  new_position;

    if (use_madgwick) {
        imu_orientation_filter_madgwick.setOrientation(msg.pose.orientation.w, 
                                                       msg.pose.orientation.x, 
                                                       msg.pose.orientation.y, 
                                                       msg.pose.orientation.z);
    } else {
        imu_orientation_filter_valenti.setOrientation( msg.pose.orientation.w, 
                                                       msg.pose.orientation.x, 
                                                       msg.pose.orientation.y, 
                                                       msg.pose.orientation.z);
    }
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
    ros::Subscriber imu_callback_node    = subscriber_node.subscribe("/uav/sensors/imu", 10, &uavIMUCallback);
    ros::Subscriber irodom_callback_node = subscriber_node.subscribe("/puffin_ir_marker_odometry/pose", 10, &irMarkerOdometryCallback);

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
