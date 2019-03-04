#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <iostream>

#include <geometry_msgs/PoseStamped.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>

#include <mav_msgs/conversions.h>
#include <mav_trajectory_generation/polynomial_optimization_nonlinear.h>
#include <mav_trajectory_generation/trajectory.h>
#include <mav_trajectory_generation/trajectory_sampling.h>
#include <mav_trajectory_generation_ros/ros_visualization.h>
#include <mav_visualization/hexacopter_marker.h>

using namespace std;
using namespace mav_trajectory_generation;


int main(int argc, char** argv)
{
    ros::init(argc, argv, "puffin_pilot");
    ros::NodeHandle node_handle;
    ros::start();


    mav_trajectory_generation::Vertex::Vector vertices;
    const int dimension = 4;
    const int derivative_to_optimize = mav_trajectory_generation::derivative_order::SNAP;
    


    mav_trajectory_generation::Vertex start(dimension), 
                                      middle1(dimension), 
                                      middle2(dimension), 
                                      middle3(dimension), 
                                      middle4(dimension), 
                                      middle5(dimension), 
                                      end(dimension);


    start.makeStartOrEnd(Eigen::Vector4d(16.5, -1, 5, 1.57), derivative_to_optimize);
    vertices.push_back(start);

    middle1.addConstraint(mav_trajectory_generation::derivative_order::POSITION, Eigen::Vector4d(17, 2.5, 7, 1.57));
    vertices.push_back(middle1);

    end.makeStartOrEnd(Eigen::Vector4d(16, 35.5, 6, 1.57), derivative_to_optimize);
    vertices.push_back(end);

/*
    middle2.addConstraint(mav_trajectory_generation::derivative_order::POSITION, Eigen::Vector4d(17, 7.5, 6, 1.57));
    vertices.push_back(middle2);

    end.makeStartOrEnd(Eigen::Vector4d(13, 10.0, 6, 1.57), derivative_to_optimize);
    vertices.push_back(end);
*/





    std::vector<double> segment_times;
    const double v_max = 10.0;
    const double a_max = 10.0;
    segment_times = estimateSegmentTimes(vertices, v_max, a_max);

    NonlinearOptimizationParameters parameters;
    parameters.max_iterations = 1000;
    parameters.f_rel = 0.05;
    parameters.x_rel = 0.1;
    parameters.time_penalty = 500.0;
    parameters.initial_stepsize_rel = 0.1;
    parameters.inequality_constraint_tolerance = 0.1;


    const int N = 10;
    PolynomialOptimizationNonLinear<N> opt(dimension, parameters);
    opt.setupFromVertices(vertices, segment_times, derivative_to_optimize);
    opt.addMaximumMagnitudeConstraint(mav_trajectory_generation::derivative_order::VELOCITY, v_max);                                
    opt.addMaximumMagnitudeConstraint(mav_trajectory_generation::derivative_order::ACCELERATION, a_max);
    opt.optimize();


    mav_trajectory_generation::Segment::Vector segments;
    opt.getPolynomialOptimizationRef().getSegments(&segments);

    mav_trajectory_generation::Trajectory trajectory;

    opt.getTrajectory(&trajectory);


    printf("ok\n");

/*
    // Single sample:
    double sampling_time = 2.0;
    int derivative_order = mav_trajectory_generation::derivative_order::POSITION;
    Eigen::VectorXd sample = trajectory.evaluate(sampling_time, derivative_order);

    // Sample range:
    double t_start = 2.0;
    double t_end = 10.0;
    double dt = 0.01;
    std::vector<Eigen::VectorXd> result;
    std::vector<double> sampling_times; // Optional.
    trajectory.evaluateRange(t_start, t_end, dt, derivative_order, &result, &sampling_times);
*/


    mav_msgs::EigenTrajectoryPoint state;
    mav_msgs::EigenTrajectoryPoint::Vector states;
    bool success;

    // Single sample:
    //double sampling_time = 2.0;
    //bool success = mav_trajectory_generation::sampleTrajectoryAtTime(trajectory, sampling_time, &state);

    // Sample range:
    //double t_start = 2.0;
    //double duration = 6.5;
    //double dt = 0.01;
    //success = mav_trajectory_generation::sampleTrajectoryInRange(trajectory, t_start, duration, dt, &states);

    // Whole trajectory:
    double sampling_interval = 0.5;
    success = mav_trajectory_generation::sampleWholeTrajectory(trajectory, sampling_interval, &states);


    visualization_msgs::MarkerArray markers;
    double distance = 1.0; // Distance by which to seperate additional markers. Set 0.0 to disable.
    std::string frame_id = "world";


    mav_trajectory_generation::drawMavTrajectory(trajectory, distance, frame_id, &markers);
    
    //bool simple = false;
    //mav_visualization::HexacopterMarker hex(simple);
    // From Trajectory class:
    //mav_trajectory_generation::drawMavTrajectoryWithMavMarker(trajectory, distance, frame_id, hex, &markers);
    // From mav_msgs::EigenTrajectoryPoint::Vector states:
    //mav_trajectory_generation::drawMavSampledTrajectoryWithMavMarker(states, distance, frame_id, hex, &markers);

    ros::Publisher vis_pub = node_handle.advertise<visualization_msgs::MarkerArray>("visualization_marker", 0);
    ros::Publisher trajectory_pub = node_handle.advertise<trajectory_msgs::MultiDOFJointTrajectory>("trajectory", 1, true);
    ros::Publisher pose_pub = node_handle.advertise<geometry_msgs::PoseStamped>("pose", 0);



    trajectory_msgs::MultiDOFJointTrajectory trajectory_msg;
    mav_msgs::msgMultiDofJointTrajectoryFromEigen(states, &trajectory_msg);
    

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
            trajectory_msg.header.stamp = ros::Time::now();
            trajectory_pub.publish(trajectory_msg);
            vis_pub.publish(markers);

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
