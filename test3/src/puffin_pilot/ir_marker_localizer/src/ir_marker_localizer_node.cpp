#include <ros/ros.h>
#include <math.h>
#include <iostream>
#include "boost/multi_array.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Geometry>
#include <std_msgs/Float64.h>
#include <nav_msgs/Odometry.h>
#include <mav_msgs/conversions.h>
#include <mav_msgs/eigen_mav_msgs.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <flightgoggles/IRMarkerArray.h>
#include <puffin_pilot/GateInfo.h>
#include "rate_limiter.h"

using namespace std;


RateLimiter *ir_beacons_rate_limiter;
RateLimiter *camera_info_rate_limiter;

cv::Mat camera_matrix(3, 3, cv::DataType<double>::type);
cv::Mat dist_coeffs  (4, 1, cv::DataType<double>::type);
bool    camera_matrix_valid;

puffin_pilot::GateInfoConstPtr gate_info;
ros::Publisher ir_marker_odometry_pose_node;
ros::Publisher ir_marker_visibility_node;

bool last_odometry_valid = false;
nav_msgs::Odometry last_odometry;


void odometry_callback(const nav_msgs::Odometry& msg)
{
    ROS_INFO_ONCE("IR marker localizer received first odometry message.");

    last_odometry = msg;
    last_odometry_valid = true;
}

void rotation_matrix_to_euler_angles(cv::Mat &R, double *roll, double *pitch, double *yaw)
{
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0));
    bool singular = sy < 1e-6; // If
    if (!singular)
    {
        *roll  = atan2( R.at<double>(2,1), R.at<double>(2,2));
        *pitch = atan2(-R.at<double>(2,0), sy);
        *yaw   = atan2( R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        *roll  = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        *pitch = atan2(-R.at<double>(2,0), sy);
        *yaw   = 0;
    }
}

Eigen::Quaterniond euler_to_quaternion( const double roll, const double pitch, const double yaw )
{
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
    return q;
}

void send_ir_visibility(double pos) {
    double visibility;

    if (pos < 0 || pos >= 768) {
        visibility = -1;
    } else if (pos > 200) {
        visibility = 1;
    } else {
        visibility = pos / 200;
    }

    std_msgs::Float64 visibility_msg;
    visibility_msg.data = visibility;
    ir_marker_visibility_node.publish(visibility_msg);
}

void ir_beacons_callback(const flightgoggles::IRMarkerArrayConstPtr& ir_beacons_array)
{
    ROS_INFO_ONCE("IR marker localizer received first IR marker array message.");

    // Limit incoming message frequency to 60 Hz.
    // However, it seems to already be limited to 20 Hz.
    //if (!ir_beacons_rate_limiter->try_aquire(1)) {
    //    return;
    //}

    // Check that there is an active gate_info.
    if (!gate_info) {
        return;
    }

    // Check that we have a valid camera_matrix.
    if (!camera_matrix_valid) {
        return;
    }

    // Get IR Beacon position on camera plane
    double min_y = 9e10;
    double ir_beacons_px[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    bool   ir_beacons_visible[4] = {false, false, false, false};
    int    ir_beacons_count = 0;
    for (flightgoggles::IRMarker ir_marker : ir_beacons_array->markers) {
        if (ir_marker.landmarkID.data.compare(gate_info->gate_name.data) != 0) {
            continue;
        }
        try {
            int ir_beacon_id = std::stoi(ir_marker.markerID.data) - 1;
            if (ir_beacon_id < 0 || ir_beacon_id > 3) {
                // ID out of range, should never happen.
                continue;
            }
            ir_beacons_px[ir_beacon_id][0] = ir_marker.x;
            ir_beacons_px[ir_beacon_id][1] = ir_marker.y;
            ir_beacons_visible[ir_beacon_id] = true;

            if (ir_marker.y < min_y) {
                min_y = ir_marker.y;
            }
        } catch (...) {
            // IR marker ID could not be parsed, something is very wrong.
        }
        ir_beacons_count++;
    }
    if (ir_beacons_count < 3) {
        // We need at least 3 corners to estimate the gate position.
        send_ir_visibility(-1);
        return;
    }

    if (!last_odometry_valid and ir_beacons_count < 4) {
        // If we don't have an extrinsic guess we need all four corners.
        send_ir_visibility(-1);
        return;
    }

    send_ir_visibility(min_y);

    bool has_extrinsic_guess = false;
    cv::Mat rotation_vector    = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
    cv::Mat translation_vector = cv::Mat::zeros(3, 1, cv::DataType<double>::type);

    if (last_odometry_valid) {
        // Calculate extrinsic guess.
        mav_msgs::EigenOdometry odometry;
        eigenOdometryFromMsg(last_odometry, &odometry);

        Eigen::Quaterniond q_imu = odometry.orientation_W_B;
        Eigen::Quaterniond q_rot = euler_to_quaternion(0,  -1.5708, 1.5708);
        Eigen::Quaterniond q_cam = q_imu * q_rot.inverse();
        Eigen::Vector3d tvec = -(q_cam.inverse() * odometry.position_W);
        translation_vector.at<double>(0) = tvec.x();
        translation_vector.at<double>(1) = tvec.y();
        translation_vector.at<double>(2) = tvec.z();

        Eigen::Matrix3d rotation_matrix = q_cam.toRotationMatrix().transpose();

        cv::Mat rotation_matrix_cv = cv::Mat::zeros(3, 3, cv::DataType<double>::type);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                rotation_matrix_cv.at<double>(i, j) = rotation_matrix(i, j);
            }
        }

        Rodrigues(rotation_matrix_cv, rotation_vector);

        has_extrinsic_guess = true;
    }

    ros::Time t_start = ros::Time::now();

    // Store all camera and gate points.
    std::vector<cv::Point2d> camera_points;
    std::vector<cv::Point3d> gate_info_points;
    for (int beacon_id = 0; beacon_id < 4; beacon_id++) {
        if (!ir_beacons_visible[beacon_id]) {
            continue;
        }

        double camera_x = ir_beacons_px[beacon_id][0];
        double camera_y = ir_beacons_px[beacon_id][1];
        camera_points.push_back(cv::Point2d(camera_x, camera_y));

        double gate_x = gate_info->ir_markers.data[beacon_id * 3 + 0];
        double gate_y = gate_info->ir_markers.data[beacon_id * 3 + 1];
        double gate_z = gate_info->ir_markers.data[beacon_id * 3 + 2];
        gate_info_points.push_back(cv::Point3d(gate_x, gate_y, gate_z));
    }

    // Estimate the camera pose.
    bool solved = false;
    if (has_extrinsic_guess) {
        cv::Mat rvec = rotation_vector.clone();
        cv::Mat tvec = translation_vector.clone();
        solved = cv::solvePnP(gate_info_points, camera_points, camera_matrix, dist_coeffs, rvec, tvec, has_extrinsic_guess, cv::SOLVEPNP_ITERATIVE);
        double error = 0;
        for (int j = 0; j < 3; j++) {
            error += pow(tvec.at<double>(j) - translation_vector.at<double>(j), 2);
            // E = ||Q1*Q2−1 - 1||2 
        }
        rotation_vector = rvec.clone();
        translation_vector = tvec.clone();
    } else {
        solved = cv::solvePnP(gate_info_points, camera_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector, has_extrinsic_guess, cv::SOLVEPNP_ITERATIVE);
    }
    if (!solved) {
        // Something went wrong, could not estimate pose!
        return;
    }

    // Calculate camera position and rotation.
    cv::Mat camera_position = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
    cv::Mat rotation_matrix = cv::Mat::zeros(3, 3, cv::DataType<double>::type);
    Rodrigues(rotation_vector, rotation_matrix);
    rotation_matrix =  rotation_matrix.t();
    camera_position = -rotation_matrix * translation_vector;
    double camera_roll, camera_pitch, camera_yaw;
    rotation_matrix_to_euler_angles(rotation_matrix, &camera_roll, &camera_pitch, &camera_yaw);

    // Broadcast the estimated camera/odometry location.
    static tf2_ros::TransformBroadcaster tf_broadcaster;
    tf2::Quaternion q_imu, q_rot;
    q_imu.setRPY(camera_roll, camera_pitch, camera_yaw);

    q_rot.setRPY(0,  -1.5708,  1.5708);
    q_imu = q_imu * q_rot;

    // Broadcast transform.
    geometry_msgs::TransformStamped tf_odom;
    tf_odom.header.stamp = ir_beacons_array->header.stamp;
    tf_odom.header.frame_id = "puffin_nest";
    tf_odom.child_frame_id  = "ir_odometry";
    tf_odom.transform.translation.x = camera_position.at<double>(0);
    tf_odom.transform.translation.y = camera_position.at<double>(1);
    tf_odom.transform.translation.z = camera_position.at<double>(2);
    tf_odom.transform.rotation.x = q_imu.x();
    tf_odom.transform.rotation.y = q_imu.y();
    tf_odom.transform.rotation.z = q_imu.z();
    tf_odom.transform.rotation.w = q_imu.w();
    tf_broadcaster.sendTransform(tf_odom);

    // Publish IR marker odometry transform for use in sensor fusion.
    geometry_msgs::PoseStamped ps_ir_marker_odometry;
    ps_ir_marker_odometry.header.stamp = ir_beacons_array->header.stamp;
    ps_ir_marker_odometry.pose.position.x = camera_position.at<double>(0);
    ps_ir_marker_odometry.pose.position.y = camera_position.at<double>(1);
    ps_ir_marker_odometry.pose.position.z = camera_position.at<double>(2);
    ps_ir_marker_odometry.pose.orientation.x = q_imu.x();
    ps_ir_marker_odometry.pose.orientation.y = q_imu.y();
    ps_ir_marker_odometry.pose.orientation.z = q_imu.z();
    ps_ir_marker_odometry.pose.orientation.w = q_imu.w();
    ir_marker_odometry_pose_node.publish(ps_ir_marker_odometry);
}

void gate_info_callback(const puffin_pilot::GateInfoConstPtr& _gate_info)
{
    ROS_INFO_ONCE("IR marker localizer got first gate info message.");

    // Store incoming gate info.
    gate_info = _gate_info;
}

void camera_info_callback(const sensor_msgs::CameraInfoConstPtr& camera_info)
{
    if (!camera_info_rate_limiter->try_aquire(1)) {
        return;
    }

    cv::setIdentity(camera_matrix);
    camera_matrix.at<double>(0,0) = camera_info->K[0]; // fx
    camera_matrix.at<double>(1,1) = camera_info->K[4]; // fy
    camera_matrix.at<double>(0,2) = camera_info->K[2]; // cx
    camera_matrix.at<double>(1,2) = camera_info->K[5]; // cy
    dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);
    camera_matrix_valid = true;
}

int main(int argc, char** argv)
{
    camera_matrix_valid = false;
    ir_beacons_rate_limiter  = new RateLimiter(60.0);
    camera_info_rate_limiter = new RateLimiter(0.1);

    ros::init(argc, argv, "ir_marker_localizer");

    ros::NodeHandle publisher_node;
    ir_marker_odometry_pose_node = publisher_node.advertise<geometry_msgs::PoseStamped>("ir_markers_pose", 1, false);
    ir_marker_visibility_node    = publisher_node.advertise<std_msgs::Float64>("ir_visibility",   1, false);

    ros::NodeHandle subscriber_node;
    ros::Subscriber camera_info_subscriber = subscriber_node.subscribe("camera_info", 10, &camera_info_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber ir_beacons_subscriber  = subscriber_node.subscribe("ir_beacons",  10, &ir_beacons_callback,  ros::TransportHints().tcpNoDelay());
    ros::Subscriber gate_info_subscriber   = subscriber_node.subscribe("gate_info",   10, &gate_info_callback,   ros::TransportHints().tcpNoDelay());
    ros::Subscriber odometry_node          = subscriber_node.subscribe("odometry",    1,  &odometry_callback,    ros::TransportHints().tcpNoDelay());

    ros::spin();
    return 0;
}
