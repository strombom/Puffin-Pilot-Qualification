#include <ros/ros.h>
#include <iostream>
#include <mutex>

#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>
#include <geometry_msgs/Vector3.h>
#include <mav_msgs/conversions.h>
#include <mav_trajectory_generation/polynomial_optimization_nonlinear.h>
#include <mav_trajectory_generation/trajectory.h>
#include <mav_trajectory_generation/trajectory_sampling.h>
#include <mav_trajectory_generation_ros/ros_visualization.h>
#include <mav_visualization/hexacopter_marker.h>
#include <geometry_msgs/PoseStamped.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>

#include <puffin_pilot/Waypoints.h>
#include <puffin_pilot/PaceNote.h>

using namespace std;
using namespace mav_trajectory_generation;

// Drone characteristics
static const double v_max = 19.0;
static const double a_max = 37.0;
static const double j_max = 10.0;

// IR status, false during IR measurement, true otherwise
bool ir_done = true; 
double ir_visibility = -1;

// Trajectory waypoints
mav_msgs::EigenTrajectoryPoint::Vector waypoints;
int waypoint_idx = 0;
bool has_waypoints  = false;

// Pace notes
vector<ros::Time> pace_note_timestamps;
vector<double> pace_note_velocities;
vector<long> pace_note_measure_ir;
double pace_note_velocity = v_max;

ros::Publisher trajectory_marker_pub;
ros::Publisher trajectory_pub;
ros::Publisher odometry_mpc_pub;
ros::Publisher ir_trig_pub;


void add_constraint(mav_trajectory_generation::Vertex *v, int derivative, Eigen::Vector3d vec3, double yaw)
{
    Eigen::Vector4d vec4;
    vec4.head<3>() = vec3;
    vec4.w() = yaw;
    v->addConstraint(derivative, vec4);
}

double get_yaw(double previous_yaw, double new_yaw)
{
    // Make sure that yaw does not wrap around at zero crossing
    while (new_yaw + M_PI < previous_yaw) new_yaw += 2 * M_PI;
    while (new_yaw - M_PI > previous_yaw) new_yaw -= 2 * M_PI;
    return new_yaw;
}

void pace_note_callback(const puffin_pilot::PaceNoteConstPtr& pace_note)
{
    ROS_INFO_ONCE("Trajectory generator received first pace note.");

    pace_note_timestamps.clear();
    pace_note_velocities.clear();
    pace_note_measure_ir.clear();

    for (int idx = 0; idx < pace_note->timestamps.layout.dim[0].size; idx++) {
        pace_note_timestamps.push_back(ros::Time::now() + ros::Duration(pace_note->timestamps.data[idx]));
        pace_note_velocities.push_back(pace_note->velocities.data[idx]);
        pace_note_measure_ir.push_back(pace_note->measure_ir.data[idx]);
    }

    ir_done = true;
}

void ir_ok_callback(const std_msgs::Bool ok_msg)
{
    ir_done = true;
}

void ir_visibility_callback(const std_msgs::Float64 visibility_msg)
{
    // IR visibility is centered around 0.5.
    // Min 0, Max 1
    // If no IR visible: -1
    ir_visibility = visibility_msg.data;
    if (ir_visibility > 1) {
        ir_visibility = 1;
    }
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

/*
#define FILTER_TAP_NUM 9

static double filter_taps[FILTER_TAP_NUM] = {
  -0.05137309583491086,
  -0.01963305870705502,
  0.10790181239553044,
  0.28270681608644244,
  0.3658199490492092,
  0.28270681608644244,
  0.10790181239553044,
  -0.01963305870705502,
  -0.05137309583491086
};
*/
/*

FIR filter designed with
http://t-filter.appspot.com

sampling frequency: 960 Hz

* 0 Hz - 100 Hz
  gain = 1
  desired ripple = 2 dB
  actual ripple = 1.1371609825038353 dB

* 240 Hz - 480 Hz
  gain = 0
  desired attenuation = -50 dB
  actual attenuation = -52.62174139873899 dB

*/

#define FILTER_TAP_NUM 13

static double filter_taps[FILTER_TAP_NUM] = {
  -0.011228693762129966,
  -0.030759089104831324,
  -0.03313654816711353,
  0.02060388775363574,
  0.1380750767988882,
  0.26618633580639356,
  0.3223042307894251,
  0.26618633580639356,
  0.1380750767988882,
  0.02060388775363574,
  -0.03313654816711353,
  -0.030759089104831324,
  -0.011228693762129966
};

static double xb[FILTER_TAP_NUM];
static double yb[FILTER_TAP_NUM];
static double zb[FILTER_TAP_NUM];

Eigen::Vector3d puffin_acc;

void acc_callback(const geometry_msgs::Vector3& acc_msg)
{   
    /*
    FIR filter designed with
    http://t-filter.appspot.com

    sampling frequency: 960 Hz

    * 0 Hz - 100 Hz
      gain = 1
      desired ripple = 1 dB
      actual ripple = 0.4473503332033286 dB

    * 240 Hz - 480 Hz
      gain = 0
      desired attenuation = -22 dB
      actual attenuation = -26.50826359079475 dB
    */

    
    static bool first = true;
    if (first) {
        for (int i = 0; i < FILTER_TAP_NUM; i++) {
            xb[i] = 0;
            yb[i] = 0;
            zb[i] = 0;
        }
        first = false;
    }

    for (int i = 0; i < FILTER_TAP_NUM - 1; i++) {
        xb[i] = xb[i+1];
        yb[i] = yb[i+1];
        zb[i] = zb[i+1];
    }

    xb[FILTER_TAP_NUM - 1] = acc_msg.x;
    yb[FILTER_TAP_NUM - 1] = acc_msg.y;
    zb[FILTER_TAP_NUM - 1] = acc_msg.z;

/*
    double x = 0;
    double y = 0;
    double z = 0;
    for (int i = 0; i < FILTER_TAP_NUM; i++) {
        x += xb[i] * filter_taps[i];
        y += yb[i] * filter_taps[i];
        z += zb[i] * filter_taps[i];
    }

    puffin_acc.x() = x;
    puffin_acc.y() = y;
    puffin_acc.z() = z;
    */
}

void odometry_callback(const nav_msgs::Odometry& odometry_msg)
{
    ROS_INFO_ONCE("Trajectory generator received first odometry message.");
    if (!has_waypoints) {
        return;
    }

    static bool started = false;
    if (!started) {
        static ros::Time first_time = ros::Time::now();
        if ((ros::Time::now() - first_time).toSec() > 0.1) {
            started = true;
        }
        return;
    }

    static ros::Time previous_time = ros::Time::now();
    double diff_time = (ros::Time::now() - previous_time).toSec();
    if (diff_time < 0.0065) {
        return;
    }
    previous_time = ros::Time::now();

    mav_msgs::EigenOdometry odometry;
    eigenOdometryFromMsg(odometry_msg, &odometry);

    if (waypoint_idx == waypoints.size()) {
        ROS_INFO_ONCE("Trajectory generator finished.");
        return;
    }

    // Find current location on trajectory
    bool moved_forward = false;
    Eigen::Vector3d plane_p;
    Eigen::Vector3d plane_n;
    for (int i = waypoint_idx + 1; i < waypoints.size(); i++) {
        plane_p = waypoints[i].position_W;
        plane_n = waypoints[i].velocity_W;
        if (plane_n.norm() < 0.01) {
            plane_n = Eigen::Vector3d(0.0, 1.0, 0.3).normalized();
        } else {
            plane_n = plane_n.normalized();
        }
        double distance = plane_n.dot(odometry.position_W - plane_p);
        if (distance >= 0.0) {
            waypoint_idx++;
            moved_forward = true;
        } else {
            break;
        }
    }
    if (!moved_forward) {
        int wp_stop = max(0, waypoint_idx - 500);
        for (int i = waypoint_idx - 1; i > wp_stop; i--) {
            plane_p = waypoints[i].position_W;
            plane_n = waypoints[i].velocity_W;
            if (plane_n.norm() < 0.01) {
                plane_n = Eigen::Vector3d(0.0, 1.0, 0.3).normalized();
            } else {
                plane_n = plane_n.normalized();
            }
            double distance = plane_n.dot(odometry.position_W - plane_p);
            if (distance < 0.0) {
                waypoint_idx--;
            } else {
                break;
            }
        }
    }
    
    // Read pace notes
    if (pace_note_timestamps.size() > 0) {
        if (ros::Time::now() > pace_note_timestamps.front()) {
            pace_note_velocity = pace_note_velocities.front();
            if (pace_note_measure_ir.front() == 1) {
                ir_done = false;
                ir_trig_pub.publish(true);
            } else {
            }
            pace_note_timestamps.erase(pace_note_timestamps.begin());
            pace_note_velocities.erase(pace_note_velocities.begin());
            pace_note_measure_ir.erase(pace_note_measure_ir.begin());
        }
    }

    // Update current velocity depending on pace notes
    static double v_desired = 20.0;
    if (!ir_done) {
        double diff = v_desired - 8;
        if (abs(diff) < 0.1) {
            v_desired = 8;
        } else {
            if (v_desired > 13) {
                v_desired -= 0.1;
            } else if (v_desired > 10) {
                v_desired -= 0.1;
            } else if (v_desired > 8) {
                v_desired -= 0.03;
            } else if (v_desired > 5) {
                v_desired -= 0.015;
            }
        }

/*
        double v_delta_max = 0.15;
        if (ir_visibility < 0) {
            if (abs(v_desired - 3) < 0.1) {
                v_desired = 3;
            } else {
                v_desired -= v_delta_max;
            }
        } else {
            double p_error = ir_visibility - 0.5;

            const double kp = 0.05;
            const double kd = 0.00;

            // Derivative
            static double add = 0.95;
            static double p_error_avg_previous = p_error;
            static double p_error_avg  = p_error;
            p_error_avg =  p_error * add + (1 - add) * p_error_avg;

            double d_error  = p_error_avg - p_error_avg_previous;
            p_error_avg_previous = p_error_avg;

            // Derivative ceiling
            double max_d_error = kp / 2;
            if (abs(d_error) > max_d_error) {
                d_error = max_d_error * d_error / abs(d_error);
            }

            double v_delta;
            v_delta = kp * p_error + kd * d_error;

            double v_delta_k = 0.05 * v_desired;
            v_delta *= v_delta_k;


            v_desired += v_delta;

            printf("v % 7.3f d % 7.3f\n", v_desired, v_delta);
        }
        */
    } else {
        double diff = v_desired - pace_note_velocity;
        if (diff != 0 && abs(diff) < 0.1) {
            v_desired = pace_note_velocity;
        } else {
            if (v_desired < pace_note_velocity) {
                if (v_desired < 12) {
                    v_desired += 0.1;
                } else {
                    v_desired += 0.1;
                }
            } else if (v_desired > pace_note_velocity) {
                v_desired -= 0.1;
            }
        }
    }
    if (v_desired > v_max) {
        v_desired = v_max;
    } else if (v_desired < 3) {
        v_desired = 3;
    }

    //v_desired = 8;

    //printf("vd % 2.6f\n", v_desired);


    // Filter acceleration
    /*
    double x = 0;
    double y = 0;
    double z = 0;
    for (int i = 0; i < FILTER_TAP_NUM; i++) {
        x += xb[i] * filter_taps[i];
        y += yb[i] * filter_taps[i];
        z += zb[i] * filter_taps[i];
    }
    puffin_acc.x() = x;
    puffin_acc.y() = y;
    puffin_acc.z() = z;
    */

    // Calculate trajectory waypoints depending on velocity
    int look_ahead[6];
    for (int i = 0; i < 6; i++) {
        look_ahead[i] = (i + 1) * (v_desired / v_max * 125);
    }

/*
    static Eigen::Vector3d vel_prev = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d vel = odometry.getVelocityWorld();
    Eigen::Vector3d acc = (vel - vel_prev) / diff_time;
    vel_prev = vel;
    */
    

    static double previous_yaw = 1.57;
    double yaws[7];
    yaws[0] = get_yaw(previous_yaw, waypoints[waypoint_idx].getYaw());
    yaws[1] = get_yaw(yaws[0], waypoints[waypoint_idx + look_ahead[0]].getYaw());
    yaws[2] = get_yaw(yaws[1], waypoints[waypoint_idx + look_ahead[1]].getYaw());
    yaws[3] = get_yaw(yaws[2], waypoints[waypoint_idx + look_ahead[2]].getYaw());
    yaws[4] = get_yaw(yaws[3], waypoints[waypoint_idx + look_ahead[3]].getYaw());
    yaws[5] = get_yaw(yaws[4], waypoints[waypoint_idx + look_ahead[4]].getYaw());
    yaws[6] = get_yaw(yaws[5], waypoints[waypoint_idx + look_ahead[5]].getYaw());
    previous_yaw = yaws[6];

    Eigen::Vector3d delta_position = odometry.position_W - waypoints[waypoint_idx].position_W;
    Eigen::Vector3d start_position = odometry.position_W;

    if (delta_position.norm() > 0.9) {
        //printf("delta! %f\n", delta_position.norm());
    } else {
        delta_position *= 0;
    }

    //start_position = odometry.position_W - delta_position * 0.2;

    static const int dimension = 4;
    static const int derivative_to_optimize = derivative_order::JERK;

    Vertex::Vector vertices;
    mav_trajectory_generation::Vertex v1(dimension), v2(dimension), v3(dimension), v4(dimension), v5(dimension), v6(dimension), v7(dimension);
    
    add_constraint(&v1, derivative_order::POSITION, start_position, yaws[0]);
    add_constraint(&v1, derivative_order::VELOCITY, odometry.getVelocityWorld(), odometry.getYawRate());
    //add_constraint(&v1, derivative_order::ACCELERATION, puffin_acc, waypoints[waypoint_idx + look_ahead[0]].getYawAcc());
    vertices.push_back(v1);

    add_constraint(&v2, derivative_order::POSITION, waypoints[waypoint_idx + look_ahead[0]].position_W + delta_position * 0.5, yaws[1]);
    vertices.push_back(v2);

    add_constraint(&v3, derivative_order::POSITION, waypoints[waypoint_idx + look_ahead[1]].position_W + delta_position * 0.3, yaws[2]);
    //add_constraint(&v3, derivative_order::VELOCITY, waypoints[waypoint_idx + look_ahead[1]].velocity_W * v_desired / v_max, waypoints[waypoint_idx + look_ahead[1]].getYawRate());
    vertices.push_back(v3);

    add_constraint(&v4, derivative_order::POSITION, waypoints[waypoint_idx + look_ahead[2]].position_W + delta_position * 0.2, yaws[3]);
    vertices.push_back(v4);

    add_constraint(&v5, derivative_order::POSITION, waypoints[waypoint_idx + look_ahead[3]].position_W + delta_position * 0.1, yaws[4]);
    vertices.push_back(v5);

    add_constraint(&v6, derivative_order::POSITION, waypoints[waypoint_idx + look_ahead[4]].position_W + delta_position * 0.05, yaws[5]);
    vertices.push_back(v6);

    add_constraint(&v7, derivative_order::POSITION,     waypoints[waypoint_idx + look_ahead[5]].position_W, yaws[6]);
    add_constraint(&v7, derivative_order::VELOCITY,     waypoints[waypoint_idx + look_ahead[5]].velocity_W * v_desired / v_max, waypoints[waypoint_idx + look_ahead[5]].getYawRate());
    add_constraint(&v7, derivative_order::ACCELERATION, waypoints[waypoint_idx + look_ahead[5]].acceleration_W, waypoints[waypoint_idx + look_ahead[5]].getYawAcc());
    vertices.push_back(v7);

    std::vector<double> segment_times;
    segment_times = estimate_segment_times(vertices, v_desired, a_max);

    for (int j = 0; j < segment_times.size(); j++) {
        segment_times[j] = 0.250;
    }

    // Initialize trajectory opitmizer
    NonlinearOptimizationParameters parameters;
    parameters.max_iterations = 40;
    parameters.use_soft_constraints = true;
    parameters.time_alloc_method = NonlinearOptimizationParameters::kMellingerOuterLoop;
    parameters.algorithm = nlopt::LD_LBFGS;
    parameters.print_debug_info = false;
    parameters.print_debug_info_time_allocation = false;

    // Optimize trajectory
    const int N = 8;
    PolynomialOptimizationNonLinear<N> opt(dimension, parameters);
    opt.setupFromVertices(vertices, segment_times, derivative_to_optimize);
    opt.addMaximumMagnitudeConstraint(derivative_order::VELOCITY, v_desired);
    opt.addMaximumMagnitudeConstraint(derivative_order::ACCELERATION, a_max);
    opt.optimize();
    mav_trajectory_generation::Trajectory trajectory;
    opt.getTrajectory(&trajectory);

    // Sample waypoints on trajectory
    double t_start = 0.0;
    double dt = 0.05;
    double duration = dt * 21;
    mav_msgs::EigenTrajectoryPoint::Vector trajectory_states;
    mav_trajectory_generation::sampleTrajectoryInRange(trajectory, t_start, duration, dt, &trajectory_states);
    trajectory_msgs::MultiDOFJointTrajectory trajectory_msg;
    mav_msgs::msgMultiDofJointTrajectoryFromEigen(trajectory_states, &trajectory_msg);
    trajectory_msg.header.stamp = ros::Time::now();

/*
    Eigen::Vector3d a = trajectory_states[0].acceleration_W;
    Eigen::Vector3d b = trajectory_states[0].velocity_W;
    Eigen::Vector3d c = puffin_acc;
    Eigen::Vector3d d = acc;
    Eigen::Vector3d e = odometry.getVelocityWorld();

    printf("a (% 5.2f % 5.2f % 5.2f) v (% 5.2f % 5.2f % 5.2f) pa (% 5.2f % 5.2f % 5.2f) oa (% 5.2f % 5.2f % 5.2f) ov (% 5.2f % 5.2f % 5.2f)\n", 
        a.x(), a.y(), a.z(), 
        b.x(), b.y(), b.z(), 
        c.x(), c.y(), c.z(), 
        d.x(), d.y(), d.z(), 
        e.x(), e.y(), e.z());

*/

    // Publish trajectory and odometry to MPC
    trajectory_pub.publish(trajectory_msg);
    odometry_mpc_pub.publish(odometry_msg);
}

std::vector<double> optimize_segment_times(std::vector<double> segment_times, const Vertex::Vector& vertices, double v_max, double a_max, double dt)
{
    int wp_idx = 0;
    for (int i = 0; i < vertices.size() - 1; ++i) {
        Eigen::VectorXd p_start, p_end;
        vertices[i].getConstraint(derivative_order::POSITION, &p_start);
        vertices[i+1].getConstraint(derivative_order::POSITION, &p_end);

        double path_length = 0;
        int wp_start = wp_idx;
        int wp_stop = wp_idx;
        // Find end of segment
        for (int j = wp_start; j < waypoints.size() - 1; j++) {
            Eigen::Vector3d pos = waypoints[j].position_W;
            Eigen::Vector3d diff = waypoints[j+1].position_W - waypoints[j].position_W;
            Eigen::Vector3d normal = diff.normalized();
            path_length += diff.norm();
            double distance = normal.dot(p_end - pos);
            if (distance >= 0.0) {
                wp_stop++;
            } else {
                break;
            }
        }

        double vel_max = 0;
        double acc_max = 0;
        for (int j = wp_start; j < wp_stop - 1; j++) {
            Eigen::Vector3d vel = waypoints[j].velocity_W;
            Eigen::Vector3d acc = waypoints[j].acceleration_W;
            if (vel.norm() > vel_max) {
                vel_max = vel.norm();
            }
            if (acc.norm() > acc_max) {
                acc_max = acc.norm();
            }
        }

        double k_vel = (vel_max / v_max - 1) * 0.91 + 1;
        double k_acc = (pow(acc_max / a_max, 0.5) - 1) * 0.09 + 1;
        double segment_time = (wp_stop - wp_start) * dt;

        if (k_vel > 1 || k_acc > 1) {
            // Too high velocity or acceleration
            segment_time *= min(k_vel, k_acc);
        } else {
            segment_time *= max(k_vel, k_acc);
        }

        segment_times[i] = segment_time * 0.99;
        wp_idx = wp_stop;
    }
    
    return segment_times;
}

void print_waypoints(const Vertex::Vector& vertices)
{
    std::vector<double> segment_times;

    int wp_idx = 0;
    for (int i = 0; i < vertices.size() - 1; ++i) {
        Eigen::VectorXd p_end;
        vertices[i+1].getConstraint(derivative_order::POSITION, &p_end);

        int wp_start = wp_idx;
        int wp_stop = wp_idx;
        for (int j = wp_start; j < waypoints.size() - 1; j++) {
            Eigen::Vector3d pos = waypoints[j].position_W;
            Eigen::Vector3d diff = waypoints[j+1].position_W - waypoints[j].position_W;
            Eigen::Vector3d normal = diff.normalized();
            double distance = normal.dot(p_end - pos);
            if (distance >= 0.0) {
                wp_stop++;
            } else {
                break;
            }
        }

        double vel_max = 0;
        double acc_max = 0;
        for (int j = wp_start; j < wp_stop - 1; j++) {
            Eigen::Vector3d vel = waypoints[j].velocity_W;
            Eigen::Vector3d acc = waypoints[j].acceleration_W;
            if (vel.norm() > vel_max) {
                vel_max = vel.norm();
            }
            if (acc.norm() > acc_max) {
                acc_max = acc.norm();
            }
        }

        //printf("vmax % 2.7f amax % 2.7f\n", vel_max, acc_max);
        wp_idx = wp_stop;
    }
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
            v.makeStartOrEnd(position, derivative_to_optimize);
        } else if (i == pn_waypoints_pos.size() - 1) {
            // Last vertex
            v.makeStartOrEnd(position, derivative_to_optimize);
        } else {
            v.addConstraint(derivative_order::POSITION, position);
        }
        vertices.push_back(v);
    }

    std::vector<double> segment_times;
    segment_times = estimate_segment_times(vertices, v_max, a_max);
    segment_times[0] *= 1.6;
    segment_times[segment_times.size()-1] *= 1.6;

    NonlinearOptimizationParameters parameters;
    parameters.max_iterations = 100;
    parameters.use_soft_constraints = true;
    parameters.time_alloc_method = NonlinearOptimizationParameters::kMellingerOuterLoop;
    parameters.algorithm = nlopt::LD_LBFGS;

    // First estimation of trajectory
    mav_trajectory_generation::Trajectory trajectory;
    const int N = 6;
    PolynomialOptimizationNonLinear<N> opt_first(dimension, parameters);
    opt_first.setupFromVertices(vertices, segment_times, derivative_to_optimize);
    opt_first.addMaximumMagnitudeConstraint(derivative_order::VELOCITY, v_max);
    opt_first.addMaximumMagnitudeConstraint(derivative_order::ACCELERATION, a_max);
    opt_first.optimize();
    opt_first.getTrajectory(&trajectory);

    double sampling_interval = 0.01;
    for (int i = 0; i < 20; i++) {
        // First estimated waypoints
        mav_trajectory_generation::sampleWholeTrajectory(trajectory, sampling_interval, &waypoints);

        // Optimize segment times
        segment_times = optimize_segment_times(segment_times, vertices, v_max, a_max, sampling_interval);

        // Optimize trajectory with updated segment times
        if (i == 19) {
            parameters.print_debug_info = true;
            parameters.print_debug_info_time_allocation = true;
        }
        PolynomialOptimizationNonLinear<N> opt(dimension, parameters);
        opt.setupFromVertices(vertices, segment_times, derivative_to_optimize);
        opt.addMaximumMagnitudeConstraint(derivative_order::VELOCITY, v_max);
        opt.addMaximumMagnitudeConstraint(derivative_order::ACCELERATION, a_max);
        opt.optimize();
        opt.getTrajectory(&trajectory);
    }

    // Visualize trajectory
    sampling_interval = 0.2;
    mav_msgs::EigenTrajectoryPoint::Vector marker_states;
    mav_trajectory_generation::sampleWholeTrajectory(trajectory, sampling_interval, &marker_states);
    visualization_msgs::MarkerArray markers;
    double distance = 0.0;
    string frame_id = "puffin_nest";
    mav_trajectory_generation::drawMavSampledTrajectory(marker_states, distance, frame_id, &markers);
    trajectory_marker_pub.publish(markers);

    // Final waypoints
    sampling_interval = 0.002;
    mav_trajectory_generation::sampleWholeTrajectory(trajectory, sampling_interval, &waypoints);

    print_waypoints(vertices);

    has_waypoints = true;
    waypoint_idx = 0;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "trajectory_generator");
    ros::NodeHandle node_handle;
    ros::start();

    trajectory_pub        = node_handle.advertise<trajectory_msgs::MultiDOFJointTrajectory>("trajectory", 1, true);
    trajectory_marker_pub = node_handle.advertise<visualization_msgs::MarkerArray>("puffin_trajectory_markers", 0);
    odometry_mpc_pub      = node_handle.advertise<nav_msgs::Odometry>("odometry_mpc", 0);
    ir_trig_pub           = node_handle.advertise<std_msgs::Bool>("ir_trig", 0);

    ros::Subscriber waypoints_subscriber  = node_handle.subscribe("waypoints",     1, &waypoints_callback);
    ros::Subscriber odometry_node         = node_handle.subscribe("odometry",      1, &odometry_callback,      ros::TransportHints().tcpNoDelay());
    //ros::Subscriber acc_node              = node_handle.subscribe("acc",           1, &acc_callback,           ros::TransportHints().tcpNoDelay());
    ros::Subscriber pace_note_node        = node_handle.subscribe("pace_note",     1, &pace_note_callback,     ros::TransportHints().tcpNoDelay());
    ros::Subscriber ir_ok_node            = node_handle.subscribe("ir_ok",         1, &ir_ok_callback,         ros::TransportHints().tcpNoDelay());
    ros::Subscriber ir_visibility_node    = node_handle.subscribe("ir_visibility", 1, &ir_visibility_callback, ros::TransportHints().tcpNoDelay());

    ros::spin();
    return 0;
}
