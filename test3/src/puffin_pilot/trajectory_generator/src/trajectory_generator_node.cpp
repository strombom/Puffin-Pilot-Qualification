#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <iostream>
#include <mutex>

#include <mav_msgs/conversions.h>
#include <mav_trajectory_generation/polynomial_optimization_nonlinear.h>
#include <mav_trajectory_generation/trajectory.h>
#include <mav_trajectory_generation/trajectory_sampling.h>
#include <mav_trajectory_generation_ros/ros_visualization.h>
#include <mav_visualization/hexacopter_marker.h>
#include <geometry_msgs/PoseStamped.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>

#include <puffin_pilot/Waypoints.h>

using namespace std;
using namespace mav_trajectory_generation;

bool has_waypoints  = false;
int current_waypoint_idx = 0;

vector<Eigen::Vector3d> waypoints_pos;
vector<Eigen::Vector3d> waypoints_vel;
vector<double> waypoints_yaw;

ros::Publisher trajectory_marker_pub;
ros::Publisher trajectory_pub;
//ros::Publisher pose_pub;
ros::Publisher odometry_mpc_pub;

static const int dimension = 4;
static const int derivative_to_optimize = derivative_order::SNAP;

mav_msgs::EigenTrajectoryPoint::Vector trajectory_states;

mutex mtx;

void odometry_callback(const nav_msgs::Odometry& odometry_msg)
{
    ROS_INFO_ONCE("Trajectory generator received first odometry message.");
    if (!has_waypoints) {
        return;
    }

    static const double v_max = 35.0;
    static const double a_max = 25.0;

    static int count = 0;
    count++;
    if (count == 8) {
        count = 0;
    } else {
        return;
    }

    static ros::WallTime previous_time = ros::WallTime::now();

    double diff = (ros::WallTime::now() - previous_time).toSec();
    previous_time = ros::WallTime::now();
    //printf("traj odom time %f\n", diff);




    ros::WallTime s1 = ros::WallTime::now();



    mav_msgs::EigenOdometry odometry;
    eigenOdometryFromMsg(odometry_msg, &odometry);

    bool new_gate = false;
    double distance = (odometry.position_W - waypoints_pos[current_waypoint_idx]).norm();
    if (distance < 1.0) {
        current_waypoint_idx++;
        ROS_INFO("New GATE PLEASE! %d\n", current_waypoint_idx);
        new_gate = true;
    }
    
    static const int TC = 3;

    Eigen::Vector4d traj_waypoints_pos[TC];
    traj_waypoints_pos[0].head<3>() = odometry.position_W;
    traj_waypoints_pos[0].w()       = waypoints_yaw[current_waypoint_idx]; // odometry.getYaw();

    Eigen::Vector4d traj_waypoints_vel[TC];
    traj_waypoints_vel[0].head<3>() = odometry.getVelocityWorld();
    traj_waypoints_vel[0].w()       = 0.0; //odometry.getYawRate();

    for (int waypoint_idx = 0; waypoint_idx < TC-1; waypoint_idx++) {
        traj_waypoints_pos[waypoint_idx + 1].head<3>() = waypoints_pos[current_waypoint_idx + waypoint_idx];
        traj_waypoints_pos[waypoint_idx + 1].w()       = waypoints_yaw[current_waypoint_idx + waypoint_idx];
        traj_waypoints_vel[waypoint_idx + 1].head<3>() = waypoints_vel[current_waypoint_idx + waypoint_idx] * 5.0;
        traj_waypoints_vel[waypoint_idx + 1].w()       = 0.0;
    }

    if (new_gate) {
        for (int idx = 0; idx < TC; idx++) {
            printf(" wp %d (%f %f %f %f)\n", idx, traj_waypoints_pos[idx].x(), traj_waypoints_pos[idx].y(), traj_waypoints_pos[idx].z(), traj_waypoints_pos[idx].w());
        }
    }

    ros::WallTime s2 = ros::WallTime::now();

    /*
    for (int waypoint_idx = 0; waypoint_idx < 4; waypoint_idx++) {
        printf(" waypoint[%d] (%f, %f, %f, %f)\n", waypoint_idx, traj_waypoints_pos[waypoint_idx].x(), traj_waypoints_pos[waypoint_idx].y(), traj_waypoints_pos[waypoint_idx].z(), traj_waypoints[waypoint_idx].w());
    }
    */

    //traj_waypoints_pos[0].w() = traj_waypoints_pos[1].w();

    Vertex::Vector vertices;
    for (int idx = 0; idx < TC; idx++) {
        mav_trajectory_generation::Vertex v(dimension);
        //printf(" add constraint %f\n", traj_waypoints_pos[idx].w());
        v.addConstraint(derivative_order::POSITION, traj_waypoints_pos[idx]);
        //v.addConstraint(derivative_order::VELOCITY, traj_waypoints_vel[idx]);

        if (idx == 0) {
            v.addConstraint(derivative_order::VELOCITY, traj_waypoints_vel[idx]);
        }
        //if (idx == 0) {
        //    Eigen::Vector4d v2;
        //    //v.addConstraint(derivative_order::VELOCITY, v2);
        //}
        vertices.push_back(v);
    }
    //printf("\n");

    std::vector<double> segment_times;
    segment_times = estimateSegmentTimes(vertices, v_max, a_max);


    static int count3 = 0;
    count3++;
    if (count3 == 100) {
        count3 = 0;
        printf("st: ");
        for (int i = 0; i < segment_times.size(); i++) {
            printf("%f ", segment_times[i]);
        }
        printf("\n");
    } else {
        //return;
    }

    ros::WallTime s3 = ros::WallTime::now();


/*
    NonlinearOptimizationParameters parameters;
    parameters.max_iterations = 100;
    parameters.f_rel = 0.05;
    parameters.x_rel = 0.1;
    parameters.time_penalty = 500.0;
    parameters.initial_stepsize_rel = 0.1;
    parameters.inequality_constraint_tolerance = 0.1;

    const int N = 10;
    PolynomialOptimizationNonLinear<N> opt(dimension, parameters);
    opt.setupFromVertices(vertices, segment_times, derivative_to_optimize);
    opt.addMaximumMagnitudeConstraint(derivative_order::VELOCITY, v_max);                                
    opt.addMaximumMagnitudeConstraint(derivative_order::ACCELERATION, a_max);
    opt.optimize();
    */

    const int N = 10;
    mav_trajectory_generation::PolynomialOptimization<N> opt(dimension);
    opt.setupFromVertices(vertices, segment_times, derivative_to_optimize);
    opt.solveLinear();



    ros::WallTime s4 = ros::WallTime::now();

    //mav_trajectory_generation::Segment::Vector segments;
    //opt.getPolynomialOptimizationRef().getSegments(&segments);

    mav_trajectory_generation::Trajectory trajectory;
    opt.getTrajectory(&trajectory);


    // Single sample:
    //double sample_time = 0.50;
    //Eigen::Vector3d pose_position = trajectory.evaluate(sampling_time, derivative_order::POSITION);
    //Eigen::Vector4d pose_orientation = trajectory.evaluate(sampling_time, derivative_order::ORIENTATION);
    //mav_msgs::EigenTrajectoryPoint state;
    //bool success = mav_trajectory_generation::sampleTrajectoryAtTime(trajectory, sample_time, &state);

    ros::WallTime s5 = ros::WallTime::now();

    /*
    geometry_msgs::PoseStamped ps;
    ps.pose.position.x = state.position_W.x();
    ps.pose.position.y = state.position_W.y();
    ps.pose.position.z = state.position_W.z();
    ps.pose.orientation.x = state.orientation_W_B.x();
    ps.pose.orientation.y = state.orientation_W_B.y();
    ps.pose.orientation.z = state.orientation_W_B.z();
    ps.pose.orientation.w = state.orientation_W_B.w();
    //printf(" set pose %f %f %f %f\n", state.orientation_W_B.x(), state.orientation_W_B.y(), state.orientation_W_B.z(), state.orientation_W_B.w());
    ps.header.stamp = ros::Time::now();
    pose_pub.publish(ps);
    */

    ros::WallTime s6 = ros::WallTime::now();


/*
    static int count2 = 499;
    count2++;
    if (count2 == 500) {
        count2 = 0;
    } else {
        return;
    }
*/

    // Sample range:
    double t_start = 0.011;
    double dt = 0.01136;
    double duration = dt * 50;
    mtx.lock();
    mav_trajectory_generation::sampleTrajectoryInRange(trajectory, t_start, duration, dt, &trajectory_states);
    mtx.unlock();

    trajectory_msgs::MultiDOFJointTrajectory trajectory_msg;
    mav_msgs::msgMultiDofJointTrajectoryFromEigen(trajectory_states, &trajectory_msg);
    trajectory_msg.header.stamp = ros::Time::now();
    trajectory_pub.publish(trajectory_msg);


    odometry_mpc_pub.publish(odometry_msg);

    return;
/*

    static int count2 = 0;
    count2++;
    if (count2 == 100) {
        count2 = 0;
    } else {
        return;
    }

    double d1 = (s2 - s1).toSec();
    double d2 = (s3 - s2).toSec();
    double d3 = (s4 - s3).toSec();
    double d4 = (s5 - s4).toSec();
    double d5 = (s6 - s5).toSec();

    printf("Trajectory time %f %f %f %f %f.\n", d1, d2, d3, d4, d5);
*/

    //return;







    //mav_msgs::EigenTrajectoryPoint state;
    //bool success;

    // Single sample:
    //double sampling_time = 2.0;
    //bool success = mav_trajectory_generation::sampleTrajectoryAtTime(trajectory, sampling_time, &state);

/*
    static double previous_end_time = 0;

    for (int i = 0; i < states.size(); i++) {
        states[i].time_from_start_ns += previous_end_time;
    }
    
    previous_end_time = previous_end_time + states[states.size()-1].time_from_start_ns + sampling_interval; //  trajectory.getMaxTime();
*/

/*
    previous_orientation.x() = states[states.size()-1].orientation_W_B.x();
    previous_orientation.y() = states[states.size()-1].orientation_W_B.y();
    previous_orientation.z() = states[states.size()-1].orientation_W_B.z();
    previous_orientation.w() = states[states.size()-1].orientation_W_B.w();
*/
    /*
    previous_velocity.x() = states[states.size()-1].velocity_W.x();
    previous_velocity.y() = states[states.size()-1].velocity_W.y();
    previous_velocity.z() = states[states.size()-1].velocity_W.z();
*/
/*
    */
    //bool simple = false;
    //mav_visualization::HexacopterMarker hex(simple);
    // From Trajectory class:
    //mav_trajectory_generation::drawMavTrajectoryWithMavMarker(trajectory, distance, frame_id, hex, &markers);
    // From mav_msgs::EigenTrajectoryPoint::Vector states:
    //mav_trajectory_generation::drawMavSampledTrajectoryWithMavMarker(states, distance, frame_id, hex, &markers);



}

void trajectory_publisher(const ros::TimerEvent& event)
{
    return;

    // Publish trajectory.
    //double sampling_interval = 0.025;
    mtx.lock();

    //mav_trajectory_generation::sampleWholeTrajectory(trajectory, sampling_interval, &states);

    visualization_msgs::MarkerArray markers;

    visualization_msgs::Marker marker;
    marker.header.frame_id = "puffin_nest";
    marker.header.stamp = ros::Time();
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.scale.x = 0.05;
    marker.color.a = 1.0;
    marker.color.g = 1.0;
    {
      for (size_t i = 0; i < trajectory_states.size(); i++) {
        geometry_msgs::Point p;
        p.x = trajectory_states[i].position_W.x();
        p.y = trajectory_states[i].position_W.y();
        p.z = trajectory_states[i].position_W.z();
        marker.points.push_back(p);
      }
    }
    markers.markers.push_back(marker);
    trajectory_marker_pub.publish(markers);

    ROS_INFO_ONCE("Trajectory generator sent trajectory markers.");

    mtx.unlock();
    return;

/*
    visualization_msgs::MarkerArray markers;
    double distance = 1.0; // Distance by which to seperate additional markers. Set 0.0 to disable.
    std::string frame_id = "world";

    mav_trajectory_generation::drawMavTrajectory(trajectory, distance, frame_id, &markers);

    for (int idx = 0; idx < markers.markers.size(); idx++) {
        markers.markers[idx].id = idx;
    }
    trajectory_marker_pub.publish(markers);
*/

    //trajectory_reference_vis_publisher_.publish(marker_queue);

}

void waypoints_callback(const puffin_pilot::WaypointsConstPtr &_waypoints)
{
    ROS_INFO("Trajectory generator received waypoints.");

    for (int idx = 0; idx < _waypoints->positions.layout.dim[0].size; idx++) {
        Eigen::Vector3d position, velocity;

        position.x() = _waypoints->positions.data[idx * _waypoints->positions.layout.dim[1].stride + 0];
        position.y() = _waypoints->positions.data[idx * _waypoints->positions.layout.dim[1].stride + 1];
        position.z() = _waypoints->positions.data[idx * _waypoints->positions.layout.dim[1].stride + 2];

        velocity.x() = _waypoints->velocities.data[idx * _waypoints->velocities.layout.dim[1].stride + 0];
        velocity.y() = _waypoints->velocities.data[idx * _waypoints->velocities.layout.dim[1].stride + 1];
        velocity.z() = _waypoints->velocities.data[idx * _waypoints->velocities.layout.dim[1].stride + 2];

        waypoints_pos.push_back(position);
        waypoints_vel.push_back(velocity);
        waypoints_yaw.push_back(_waypoints->yaws.data[idx]);

        printf(" wp %d (%f %f %f, %f %f %f, %f)\n", idx,
            waypoints_pos[idx].x(), waypoints_pos[idx].y(), waypoints_pos[idx].z(),
            waypoints_vel[idx].x(), waypoints_vel[idx].y(), waypoints_vel[idx].z(),
            waypoints_yaw[idx]);
    }
    has_waypoints = true;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "trajectory_generator");
    ros::NodeHandle node_handle;
    ros::start();

    trajectory_pub        = node_handle.advertise<trajectory_msgs::MultiDOFJointTrajectory>("trajectory", 1, true);
    trajectory_marker_pub = node_handle.advertise<visualization_msgs::MarkerArray>("puffin_trajectory_markers", 0);
    odometry_mpc_pub      = node_handle.advertise<nav_msgs::Odometry>("odometry_mpc", 0);
    //pose_pub            = node_handle.advertise<geometry_msgs::PoseStamped>("pose", 0);
    
    ros::Timer trajectory_publisher_timer = node_handle.createTimer(ros::Duration(0.5), trajectory_publisher);
    ros::Subscriber waypoints_subscriber  = node_handle.subscribe("waypoints", 1, &waypoints_callback);
    ros::Subscriber odometry_node         = node_handle.subscribe("odometry",  1, &odometry_callback,  ros::TransportHints().tcpNoDelay());

    ros::spin();

    ros::AsyncSpinner spinner(2);
    spinner.start();
    ros::waitForShutdown();



    return 0;



    geometry_msgs::PoseStamped ps;
    ps.pose.position.x = 16.0;
    ps.pose.position.y = -1.0;
    ps.pose.position.z = 6.0;
    ps.pose.orientation.x = 0;
    ps.pose.orientation.y = 0;
    ps.pose.orientation.z = 0.70710678;
    ps.pose.orientation.w = 0.70710678;
    //ps.pose.orientation.z = 0.0;
    //ps.pose.orientation.w = 1.0;
    //ps.pose.orientation.z = 1.0;
    //ps.pose.orientation.w = 0.0;

    ps.pose.orientation.x = 0;
    ps.pose.orientation.y = 0;
    ps.pose.orientation.z = 0.383;
    ps.pose.orientation.w = 0.924;

    ps.pose.orientation.x = 0;
    ps.pose.orientation.y = 0;
    ps.pose.orientation.z = 0.70710678;
    ps.pose.orientation.w = 0.70710678;
/*
    ps.pose.orientation.x = 0;
    ps.pose.orientation.y = 0;
    ps.pose.orientation.z = 0.0;
    ps.pose.orientation.w = 1.0;
*/

    int count = 0;


    ros::Rate loop_rate(1000);
    while (ros::ok()) {

/*
        if (count == 100) {
            ps.header.stamp = ros::Time::now();
            ps.pose.position.x = 16.0;
            ps.pose.position.y = -1.0;
            ps.pose.position.z = 6.0;
            ps.pose.orientation.x =  0.125;
            ps.pose.orientation.y = -0.083;
            ps.pose.orientation.z =  0.628;
            ps.pose.orientation.w =  0.764;
            pose_pub.publish(ps);
            count = 0;
        }
*/
/*
        if (count == 10) {
            ps.header.stamp = ros::Time::now();
            ps.pose.position.x = 16.0;
            ps.pose.position.y = -1.0;
            ps.pose.orientation.x = 0;
            ps.pose.orientation.y = 0;
            ps.pose.orientation.z = 0.0;
            ps.pose.orientation.w = 1.0;
            pose_pub.publish(ps);

        } else if (count == 5000) {
            ps.header.stamp = ros::Time::now();
            ps.pose.position.x = 14.0;
            ps.pose.position.y = -1.0;
            ps.pose.orientation.x = 0;
            ps.pose.orientation.y = 0;
            ps.pose.orientation.z = 0.70710678;
            ps.pose.orientation.w = 0.70710678;
            pose_pub.publish(ps);

        } else if (count == 10000) {
            ps.header.stamp = ros::Time::now();
            ps.pose.position.x = 14.0;
            ps.pose.position.y = 1.0;
            ps.pose.orientation.x = 0;
            ps.pose.orientation.y = 0;
            ps.pose.orientation.z = 0.0;
            ps.pose.orientation.w = 1.0;
            pose_pub.publish(ps);

        } else if (count == 15000) {
            ps.header.stamp = ros::Time::now();
            ps.pose.position.x = 16.0;
            ps.pose.position.y = 1.0;
            ps.pose.orientation.x = 0;
            ps.pose.orientation.y = 0;
            ps.pose.orientation.z = 0.70710678;
            ps.pose.orientation.w = 0.70710678;
            pose_pub.publish(ps);

        } else if (count == 20000) {
            count = 0;
        }*/
        count++;


        if (count == 1000) { //} || count == 2000 || count == 5000 || count == 10000) {
            //trajectory_msg.header.stamp = ros::Time::now();
            //trajectory_pub.publish(trajectory_msg);
            //vis_pub.publish(markers);

        } else if (count == 10000) {
            //count = 0;
        }



        ros::spinOnce();
        loop_rate.sleep();
    }




/*



    geometry_msgs::PoseStamped ps;
    ps.pose.position.x = 16.0;
    ps.pose.position.y = -1.0;
    ps.pose.position.z = 6.0;
    ps.pose.orientation.x = 0;
    ps.pose.orientation.y = 0;
    ps.pose.orientation.z = 0;
    ps.pose.orientation.w = 1;

    int count = 0;


    ros::Rate loop_rate(1000);
    while (ros::ok()) {
        count++;

        if (count == 1) {
            ps.pose.orientation.z = 0;
            ps.pose.orientation.w = 1;
            ps.pose.position.x = 16.5;

        } else if (count == 2000) {
            ps.pose.orientation.z = 0.7071067811865476;
            ps.pose.orientation.w = 0.7071067811865476;
            ps.pose.position.x = 16.5;


        } else if (count == 4000) {
            count = 0;
        } 

        ps.header.stamp = ros::Time::now();
        pose_pub.publish(ps);

        //vis_pub.publish(markers);
        ros::spinOnce();
        loop_rate.sleep();
    }
*/


    //printf("%d\n", (int) markers.markers.size());
/*
    for(int i = 0; i < markers.markers.size(); i++) {
        printf("marker(%d) %f %f %f, %f %f %f\n", i, markers.markers[i].points[0].x,
                                                     markers.markers[i].points[0].y,
                                                     markers.markers[i].points[0].z,
                                                     markers.markers[i].points[1].x,
                                                     markers.markers[i].points[1].y,
                                                     markers.markers[i].points[1].z);
    }


    printf("done?\n");
*/
    //ros::spin();
}
