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

ros::Publisher trajectory_marker_pub;
ros::Publisher trajectory_pub;
//ros::Publisher pose_pub;
ros::Publisher odometry_mpc_pub;

mav_msgs::EigenTrajectoryPoint::Vector waypoints;
int waypoint_idx = 0;


static const double v_max = 22.0; //22.0;
static const double a_max = 35.0; //35.0;
static const double j_max = 10.0;

void add_constraint(mav_trajectory_generation::Vertex *v, int derivative, Eigen::Vector3d vec3, double yaw)
{
    Eigen::Vector4d vec4;
    vec4.head<3>() = vec3;
    vec4.w() = yaw;
    v->addConstraint(derivative, vec4);
}


std::vector<double> estimate_segment_times(const Vertex::Vector& vertices, double v_max, double a_max)
{
    std::vector<double> segment_times;
    for (int i = 1; i < vertices.size(); ++i) {
        Eigen::VectorXd p0, p1;
        vertices[i-1].getConstraint(derivative_order::POSITION, &p0);
        vertices[i].getConstraint(derivative_order::POSITION, &p1);
        double distance = (p1 - p0).norm();
        segment_times.push_back(distance / v_max * 0.7);
    }
    return segment_times;
}

double get_yaw(double previous_yaw, double new_yaw)
{
    while (new_yaw + M_PI < previous_yaw) new_yaw += 2 * M_PI;
    while (new_yaw - M_PI > previous_yaw) new_yaw -= 2 * M_PI;
    return new_yaw;
}

void odometry_callback(const nav_msgs::Odometry& odometry_msg)
{
    ROS_INFO_ONCE("Trajectory generator received first odometry message.");
    if (!has_waypoints) {
        return;
    }

    mav_msgs::EigenOdometry odometry;
    eigenOdometryFromMsg(odometry_msg, &odometry);

    if (waypoint_idx == waypoints.size()) {
        ROS_INFO_ONCE("Trajectory generator finished.");
        return;
    }

    Eigen::Vector3d plane_p = odometry.position_W;
    Eigen::Vector3d plane_n = odometry.getVelocityWorld();
    if (plane_n.norm() > 0.1) {
        plane_n = plane_n.normalized();
    } else {
        plane_n = (odometry.orientation_W_B * Eigen::Vector3d(1.0, 0.0, 0.0)).normalized();
    }

    for (int i = waypoint_idx; i < waypoints.size(); i++) {
        // Check if we are past the next goal point
        double distance = plane_n.dot(plane_p - waypoints[i].position_W);

        //printf(" wps %d %f\n", i, distance);

        if (distance >= 0.0) {
            waypoint_idx++;
        } else {
            break;
        }
    }

    static const int look_ahead[3] = {100, 200, 300};

    static double previous_yaw = 1.57;
    double yaws[4];
    //yaws[0] = get_yaw(previous_yaw, odometry.getYaw());
    yaws[0] = get_yaw(previous_yaw, waypoints[waypoint_idx].getYaw());
    yaws[1] = get_yaw(yaws[0], waypoints[waypoint_idx + look_ahead[0]].getYaw());
    yaws[2] = get_yaw(yaws[1], waypoints[waypoint_idx + look_ahead[1]].getYaw());
    yaws[3] = get_yaw(yaws[2], waypoints[waypoint_idx + look_ahead[2]].getYaw());
    previous_yaw = yaws[3];

    static const int dimension = 4;
    static const int derivative_to_optimize = derivative_order::JERK;

    Vertex::Vector vertices;
    mav_trajectory_generation::Vertex v1(dimension), v2(dimension), v3(dimension), v4(dimension);
    
    add_constraint(&v1, derivative_order::POSITION, odometry.position_W, yaws[0]);
    add_constraint(&v1, derivative_order::VELOCITY, odometry.getVelocityWorld(), odometry.getYawRate());
    vertices.push_back(v1);

    add_constraint(&v2, derivative_order::POSITION, waypoints[waypoint_idx + look_ahead[0]].position_W, yaws[1]);
    vertices.push_back(v2);

    add_constraint(&v3, derivative_order::POSITION, waypoints[waypoint_idx + look_ahead[1]].position_W, yaws[2]);
    vertices.push_back(v3);

    add_constraint(&v4, derivative_order::POSITION,     waypoints[waypoint_idx + look_ahead[2]].position_W, yaws[3]);
    add_constraint(&v4, derivative_order::VELOCITY,     waypoints[waypoint_idx + look_ahead[2]].velocity_W, waypoints[waypoint_idx + look_ahead[2]].getYawRate());
    add_constraint(&v4, derivative_order::ACCELERATION, waypoints[waypoint_idx + look_ahead[2]].acceleration_W, waypoints[waypoint_idx + look_ahead[2]].getYawAcc());
    add_constraint(&v4, derivative_order::JERK,         waypoints[waypoint_idx + look_ahead[2]].jerk_W, 0.0);
    vertices.push_back(v4);

/*
    printf(" waypointsc %d %d\n", waypoint_idx, (int)waypoints.size());
    printf(" start  p(% 7.2f % 7.2f % 7.2f)\n", odometry.position_W.x(), odometry.position_W.y(), odometry.position_W.z());
    printf(" middle p(% 7.2f % 7.2f % 7.2f)\n", waypoints[waypoint_idx + 100].position_W.x(), waypoints[waypoint_idx + 100].position_W.y(), waypoints[waypoint_idx + 100].position_W.z());
    printf(" middle p(% 7.2f % 7.2f % 7.2f)\n", waypoints[waypoint_idx + 200].position_W.x(), waypoints[waypoint_idx + 200].position_W.y(), waypoints[waypoint_idx + 200].position_W.z());
    printf(" end    p(% 7.2f % 7.2f % 7.2f)\n", waypoints[waypoint_idx + 300].position_W.x(), waypoints[waypoint_idx + 300].position_W.y(), waypoints[waypoint_idx + 300].position_W.z());
*/



    std::vector<double> segment_times;
    segment_times = estimate_segment_times(vertices, v_max, a_max);

    //printf("segment times: ");
    for (int j = 0; j < segment_times.size(); j++) {
        //printf(" % 7.2f", segment_times[j]);
        segment_times[j] = 1.0; // *= 4.0;
    }
    //printf("\n");

    NonlinearOptimizationParameters parameters;
    //parameters.max_iterations = 100;
    //parameters.f_rel = -1;
    //parameters.time_penalty = 500.0;
    parameters.use_soft_constraints = true;
    //parameters.initial_stepsize_rel = 0.1;
    parameters.time_alloc_method = NonlinearOptimizationParameters::kMellingerOuterLoop;
    parameters.algorithm = nlopt::LD_LBFGS;
    //parameters.time_alloc_method = NonlinearOptimizationParameters::kSquaredTimeAndConstraints;
    //parameters.algorithm = nlopt::LN_BOBYQA;
    parameters.print_debug_info = false;
    parameters.print_debug_info_time_allocation = false;

    const int N = 10;
    PolynomialOptimizationNonLinear<N> opt(dimension, parameters);
    opt.setupFromVertices(vertices, segment_times, derivative_to_optimize);
    opt.addMaximumMagnitudeConstraint(derivative_order::VELOCITY, v_max);
    opt.addMaximumMagnitudeConstraint(derivative_order::ACCELERATION, a_max);
    opt.optimize();
    mav_trajectory_generation::Trajectory trajectory;
    opt.getTrajectory(&trajectory);

    // Sample range:
    double t_start = 0.0;
    double dt = 0.1;
    double duration = dt * 22;
    mav_msgs::EigenTrajectoryPoint::Vector trajectory_states;
    mav_trajectory_generation::sampleTrajectoryInRange(trajectory, t_start, duration, dt, &trajectory_states);

    trajectory_msgs::MultiDOFJointTrajectory trajectory_msg;
    mav_msgs::msgMultiDofJointTrajectoryFromEigen(trajectory_states, &trajectory_msg);
    trajectory_msg.header.stamp = ros::Time::now();

    trajectory_pub.publish(trajectory_msg);
    odometry_mpc_pub.publish(odometry_msg);
}

void waypoints_callback(const puffin_pilot::WaypointsConstPtr &_waypoints)
{
    ROS_INFO("Trajectory generator received waypoints.");

    vector<Eigen::Vector3d> pn_waypoints_pos;
    vector<double> pn_waypoints_yaw;
    for (int idx = 0; idx < _waypoints->positions.layout.dim[0].size; idx++) {
        Eigen::Vector3d position, velocity;
        position.x() = _waypoints->positions.data[idx * _waypoints->positions.layout.dim[1].stride + 0];
        position.y() = _waypoints->positions.data[idx * _waypoints->positions.layout.dim[1].stride + 1];
        position.z() = _waypoints->positions.data[idx * _waypoints->positions.layout.dim[1].stride + 2];
        pn_waypoints_pos.push_back(position);
        pn_waypoints_yaw.push_back(_waypoints->yaws.data[idx]);
        printf(" wp %d (%f %f %f, %f)\n", idx,
            pn_waypoints_pos[idx].x(), pn_waypoints_pos[idx].y(), pn_waypoints_pos[idx].z(),
            pn_waypoints_yaw[idx]);
    }


    static const int dimension = 4;
    static const int derivative_to_optimize = derivative_order::ACCELERATION;


    Vertex::Vector vertices;
    for (int i = 0; i < pn_waypoints_pos.size(); i++) {
        mav_trajectory_generation::Vertex v(dimension);
        Eigen::Vector4d position;
        position.head<3>() = pn_waypoints_pos[i];
        position.w()       = pn_waypoints_yaw[i];

        if (i == 0) {
            // First vertex
            printf("First vertex %d\n", i);
            v.makeStartOrEnd(position, derivative_to_optimize);

        } else if (i == pn_waypoints_pos.size() - 1) {
            // Last vertex
            v.makeStartOrEnd(position, derivative_to_optimize);
            printf("Last vertex %d\n", i);

        } else {
            v.addConstraint(derivative_order::POSITION, position);
            printf("Middle one %d\n", i);
        }
        vertices.push_back(v);
    }

    std::vector<double> segment_times;
    segment_times = estimate_segment_times(vertices, v_max, a_max);

    printf("segment times: ");
    for (int j = 0; j < segment_times.size(); j++) {
        printf(" % 7.2f", segment_times[j]);
        segment_times[j] *= 4.0;
    }
    printf("\n");

    NonlinearOptimizationParameters parameters;
    //parameters.max_iterations = 100;
    //parameters.f_rel = -1;
    //parameters.time_penalty = 500.0;
    parameters.use_soft_constraints = false;
    //parameters.initial_stepsize_rel = 0.1;
    parameters.time_alloc_method = NonlinearOptimizationParameters::kMellingerOuterLoop;
    parameters.algorithm = nlopt::LD_LBFGS;
    //parameters.time_alloc_method = NonlinearOptimizationParameters::kSquaredTimeAndConstraints;
    //parameters.algorithm = nlopt::LN_BOBYQA;
    parameters.print_debug_info = false;
    parameters.print_debug_info_time_allocation = false;

    const int N = 10;
    PolynomialOptimizationNonLinear<N> opt(dimension, parameters);
    opt.setupFromVertices(vertices, segment_times, derivative_to_optimize);
    opt.addMaximumMagnitudeConstraint(derivative_order::VELOCITY, v_max);
    opt.addMaximumMagnitudeConstraint(derivative_order::ACCELERATION, a_max);
    opt.optimize();
    mav_trajectory_generation::Trajectory trajectory;
    opt.getTrajectory(&trajectory);

    // Visualize trajectory
    double sampling_interval = 0.2;
    mav_msgs::EigenTrajectoryPoint::Vector marker_states;
    mav_trajectory_generation::sampleWholeTrajectory(trajectory, sampling_interval, &marker_states);
    visualization_msgs::MarkerArray markers;
    double distance = 0.0;
    string frame_id = "puffin_nest";
    mav_trajectory_generation::drawMavSampledTrajectory(marker_states, distance, frame_id, &markers);
    trajectory_marker_pub.publish(markers);

    // Waypoints
    sampling_interval = 0.01;
    mav_trajectory_generation::sampleWholeTrajectory(trajectory, sampling_interval, &waypoints);

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

    ros::Subscriber waypoints_subscriber  = node_handle.subscribe("waypoints", 1, &waypoints_callback);
    ros::Subscriber odometry_node         = node_handle.subscribe("odometry",  1, &odometry_callback,  ros::TransportHints().tcpNoDelay());

    ros::spin();
    return 0;
}
