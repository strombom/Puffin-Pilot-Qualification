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

static bool imu_initialized = false;
static ros::Time imu_last_timestamp;

struct StateEstimate {
    tf2::Quaternion angular_orientation;
    tf2::Vector3    linear_position;
    tf2::Vector3    linear_velocity;
    tf2::Vector3    linear_acceleration;
};

StateEstimate puffin_state;
const static tf2::Vector3 gravity(0.0, 0.0, 9.81);

tf2::Quaternion hamiltonToTFQuaternion(double q0, double q1, double q2, double q3)
{
  // ROS uses the Hamilton quaternion convention (q0 is the scalar). However,
  // the ROS quaternion is in the form [x, y, z, w], with w as the scalar.
  return tf2::Quaternion(q1, q2, q3, q0);
}

void uavIMUCallback(const sensor_msgs::ImuConstPtr &msg) {
    double delta_time;
    ros::Time timestamp = msg->header.stamp;

    if (!imu_initialized) {
        // First timestamp.
        imu_last_timestamp = timestamp;

        // Initialize orientation filter.        
        imu_orientation_filter_madgwick.setOrientation(puffin_state.angular_orientation.w(), 
                                                       puffin_state.angular_orientation.x(), 
                                                       puffin_state.angular_orientation.y(), 
                                                       puffin_state.angular_orientation.z());
        imu_orientation_filter_valenti. setOrientation(puffin_state.angular_orientation.w(), 
                                                       puffin_state.angular_orientation.x(), 
                                                       puffin_state.angular_orientation.y(), 
                                                       puffin_state.angular_orientation.z());
        imu_initialized = true;
        return;
    }

    delta_time = (timestamp - imu_last_timestamp).toSec();
    imu_last_timestamp = timestamp;
    if (delta_time <= 0) {
        // IMU callback arrived in wrong order.
        return;
    }

    // Update orientation
    double qx, qy, qz, qw;
    if (true) {
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

    // Update state
    StateEstimate state_new;
    tf2::Vector3 world_linear_acceleration;
    state_new.angular_orientation = tf2::Quaternion(qx, qy, qz, qw);
    state_new.linear_acceleration = tf2::Vector3(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
    world_linear_acceleration     = tf2::Transform(state_new.angular_orientation) * state_new.linear_acceleration;
    state_new.linear_velocity     = puffin_state.linear_velocity + (world_linear_acceleration - gravity) * delta_time;
    state_new.linear_position     = puffin_state.linear_position + state_new.linear_velocity * delta_time;

    // Save new state
    puffin_state = state_new;

    geometry_msgs::TransformStamped tf_imu;
    tf_imu.header.stamp = msg->header.stamp;
    tf_imu.header.frame_id = "puffin_nest";
    tf_imu.child_frame_id  = "puffin_imu";

    tf_imu.transform.translation.x = puffin_state.linear_position.x();
    tf_imu.transform.translation.y = puffin_state.linear_position.y();
    tf_imu.transform.translation.z = puffin_state.linear_position.z();

    tf_imu.transform.rotation.w = puffin_state.angular_orientation.w();
    tf_imu.transform.rotation.x = puffin_state.angular_orientation.x();
    tf_imu.transform.rotation.y = puffin_state.angular_orientation.y();
    tf_imu.transform.rotation.z = puffin_state.angular_orientation.z();
    
    static tf2_ros::TransformBroadcaster tf_broadcaster;
    tf_broadcaster.sendTransform(tf_imu);

    
    static int count = 0;
    count++;
    if (count % 20 == 0) {
        
        printf("ADV: x(% 6.2f) y(% 6.2f) z(% 6.2f) --- x(% 6.2f) y(% 6.2f) z(% 6.2f) --- x(% 6.2f) y(% 6.2f) z(% 6.2f) --- x(% 6.2f) y(% 6.2f) z(% 6.2f) --- %f\n", 
            state_new.linear_acceleration.x(), state_new.linear_acceleration.y(), state_new.linear_acceleration.z(),
            world_linear_acceleration.x(), world_linear_acceleration.y(), world_linear_acceleration.z(),
            state_new.linear_velocity.x(), state_new.linear_velocity.y(), state_new.linear_velocity.z(),
            state_new.linear_position.x(), state_new.linear_position.y(), state_new.linear_position.z(),
            delta_time);
        
        count = 0;        
    }
    
}

void irMarkerOdometryCallback(const geometry_msgs::PoseStamped& pose_msg)
{
    puffin_state.linear_position = tf2::Vector3(pose_msg.pose.position.x,
                                                pose_msg.pose.position.y,
                                                pose_msg.pose.position.z);

    puffin_state.angular_orientation = tf2::Quaternion(pose_msg.pose.orientation.x, 
                                                       pose_msg.pose.orientation.y, 
                                                       pose_msg.pose.orientation.z, 
                                                       pose_msg.pose.orientation.w);

    imu_orientation_filter_madgwick.setOrientation(pose_msg.pose.orientation.w, 
                                                   pose_msg.pose.orientation.x, 
                                                   pose_msg.pose.orientation.y, 
                                                   pose_msg.pose.orientation.z);
    imu_initialized = false;
    //imu_last_timestamp = pose_msg.header.stamp;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "puffin_state_estimation");

    std::vector<double> initial_pose;
    ros::param::get("/uav/flightgoggles_uav_dynamics/init_pose", initial_pose);

    // Orientation filter setup.
    imu_orientation_filter_valenti.setGainAcc(0.1);
    imu_orientation_filter_valenti.setDoBiasEstimation(false);
    imu_orientation_filter_valenti.setDoAdaptiveGain(false);

    imu_orientation_filter_madgwick.setAlgorithmGain(0.0);
    imu_orientation_filter_madgwick.setWorldFrame(WorldFrame::NWU);

    // Set initial state.
    puffin_state.angular_orientation = tf2::Quaternion(initial_pose.at(3), initial_pose.at(4), initial_pose.at(5), initial_pose.at(6));
    puffin_state.linear_position     = tf2::Vector3(initial_pose.at(0), initial_pose.at(1), initial_pose.at(2));
    puffin_state.linear_velocity     = tf2::Vector3(0.0, 0.0, 0.0);
    puffin_state.linear_acceleration = gravity;

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
