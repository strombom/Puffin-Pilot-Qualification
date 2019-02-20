#include <ros/ros.h>
#include <puffin_pace_notes/PaceNote.h>
#include <flightgoggles/IRMarkerArray.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <sensor_msgs/CameraInfo.h>
#include "boost/multi_array.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <math.h>

#include "ir_marker_odometry.h"
#include "rate_limiter.h"

using namespace std;


RateLimiter *ir_beacons_rate_limiter;
RateLimiter *camera_info_rate_limiter;

cv::Mat camera_matrix(3, 3, cv::DataType<double>::type);
cv::Mat dist_coeffs  (4, 1, cv::DataType<double>::type);
bool    camera_matrix_valid;

// This is information about the gate we are whizzing towards.
puffin_pace_notes::PaceNoteConstPtr pace_note;


void irBeaconsCallback(const flightgoggles::IRMarkerArrayConstPtr& ir_beacons_array)
{
    // Limit incoming message frequency to 60 Hz.
    // However, it seems to already be limited to 20 Hz.
    if (!ir_beacons_rate_limiter->try_aquire(1)) {
        //return;
    }

    // Check that there is an active pace_note.
    if (!pace_note) {
        return;
    }

    // Check that we have a valid camera_matrix.
    if (!camera_matrix_valid) {
        return;
    }

    // Get IR Beacon position on camera plane
    double ir_beacons_px[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    bool   ir_beacons_visible[4] = {false, false, false, false};
    int    ir_beacons_count = 0;
    for (flightgoggles::IRMarker ir_marker : ir_beacons_array->markers) {
        if (ir_marker.landmarkID.data.compare(pace_note->gate_name.data) != 0) {
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
        } catch (...) {
            // IR marker ID could not be parsed, something is very wrong.
        }
        ir_beacons_count++;
    }
    if (ir_beacons_count != 4) {
        // We need all 4 corners to estimate the gate position.
        return;
    }

    // Store all camera and gate points
    std::vector<cv::Point2d> camera_points;
    std::vector<cv::Point3d> pace_note_points;
    for (int beacon_id = 0; beacon_id < 4; beacon_id++) {
        double camera_x = ir_beacons_px[beacon_id][0];
        double camera_y = ir_beacons_px[beacon_id][1];
        camera_points.push_back(cv::Point2d(camera_x, camera_y));

        double gate_x = pace_note->gate_corners.data[beacon_id * 3 + 0];
        double gate_y = pace_note->gate_corners.data[beacon_id * 3 + 1];
        double gate_z = pace_note->gate_corners.data[beacon_id * 3 + 2];
        pace_note_points.push_back(cv::Point3d(gate_x, gate_y, gate_z));
    }

    // Estimate the camera pose
    cv::Mat rotation_vector    = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
    cv::Mat translation_vector = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
    bool solved = cv::solvePnP(pace_note_points, camera_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);
    if (!solved) {
        // Something went wrong, could not estimate pose!
        return;
    }

    cv::Mat camera_position = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
    cv::Mat rotation_matrix = cv::Mat::zeros(3, 3, cv::DataType<double>::type);
    Rodrigues(rotation_vector, rotation_matrix);
    rotation_matrix = rotation_matrix.t();
    camera_position = -rotation_matrix * translation_vector;

    double roll  = -atan2(-rotation_matrix.at<double>(2,1), rotation_matrix.at<double>(2,2));
    double pitch = -asin ( rotation_matrix.at<double>(2,0));
    double yaw   = -atan2(-rotation_matrix.at<double>(1,0), rotation_matrix.at<double>(0,0));

    static tf2_ros::TransformBroadcaster tf_broadcaster;
    geometry_msgs::TransformStamped tf_puffin_pose;
    tf2::Quaternion q;

    tf_puffin_pose.header.stamp = ros::Time::now();
    tf_puffin_pose.header.frame_id = "puffin_nest";
    tf_puffin_pose.child_frame_id = "ir_camera";
    tf_puffin_pose.transform.translation.x = camera_position.at<double>(0);
    tf_puffin_pose.transform.translation.y = camera_position.at<double>(1);
    tf_puffin_pose.transform.translation.z = camera_position.at<double>(2);
    q.setRPY(roll, pitch, yaw);
    tf_puffin_pose.transform.rotation.x = q.x();
    tf_puffin_pose.transform.rotation.y = q.y();
    tf_puffin_pose.transform.rotation.z = q.z();
    tf_puffin_pose.transform.rotation.w = q.w();
    tf_broadcaster.sendTransform(tf_puffin_pose);

    tf_puffin_pose.header.stamp = ros::Time::now();
    tf_puffin_pose.header.frame_id = "ir_camera";
    tf_puffin_pose.child_frame_id = "ir_odometry";
    tf_puffin_pose.transform.translation.x = 0.0;
    tf_puffin_pose.transform.translation.y = 0.0;
    tf_puffin_pose.transform.translation.z = 0.0;
    q.setRPY(0, -1.57, 1.57);
    tf_puffin_pose.transform.rotation.x = q.x();
    tf_puffin_pose.transform.rotation.y = q.y();
    tf_puffin_pose.transform.rotation.z = q.z();
    tf_puffin_pose.transform.rotation.w = q.w();
    tf_broadcaster.sendTransform(tf_puffin_pose);

    tf_puffin_pose.header.stamp = ros::Time::now();
    tf_puffin_pose.header.frame_id = "world";
    tf_puffin_pose.child_frame_id = "puffin_nest";
    tf_puffin_pose.transform.translation.x = 0;
    tf_puffin_pose.transform.translation.y = 0;
    tf_puffin_pose.transform.translation.z = 0;
    q.setRPY(0, 0, 0);
    tf_puffin_pose.transform.rotation.x = q.x();
    tf_puffin_pose.transform.rotation.y = q.y();
    tf_puffin_pose.transform.rotation.z = q.z();
    tf_puffin_pose.transform.rotation.w = q.w();
    tf_broadcaster.sendTransform(tf_puffin_pose);
}

void paceNotesCallback(const puffin_pace_notes::PaceNoteConstPtr& _pace_note)
{
    // Store incoming pace notes.
    pace_note = _pace_note;
}


void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& camera_info)
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
    ir_beacons_rate_limiter  = new RateLimiter(10.0);
    camera_info_rate_limiter = new RateLimiter(0.1);

    ros::init(argc, argv, "ir_marker_odometry");
    ros::NodeHandle subscriber_node;
    ros::Subscriber ir_beacons_subscriber  = subscriber_node.subscribe("/uav/camera/left/ir_beacons",  10, &irBeaconsCallback);
    ros::Subscriber pace_notes_subscriber  = subscriber_node.subscribe("/pace_notes/pace_note",        10, &paceNotesCallback);
    ros::Subscriber camera_info_subscriber = subscriber_node.subscribe("/uav/camera/left/camera_info", 10, &cameraInfoCallback);
    ros::spin();

    return 0;
}
