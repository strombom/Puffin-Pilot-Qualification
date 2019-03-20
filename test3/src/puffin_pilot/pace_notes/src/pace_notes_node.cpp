#include <ros/ros.h>
#include <cmath>
#include <iostream>
#include <Eigen/Geometry>
#include <visualization_msgs/Marker.h>
#include <nav_msgs/Odometry.h>
#include <mav_msgs/conversions.h>
#include <geometry_msgs/Point.h>
#include <eigen_conversions/eigen_msg.h>
#include <std_msgs/Float64MultiArray.h>

#include <puffin_pilot/GateInfo.h>
#include <puffin_pilot/PaceNote.h>
#include <puffin_pilot/Waypoints.h>

using namespace std;


int gate_count;

struct Gate {
  string name;
  Eigen::Vector3d normal;
  Eigen::Vector3d corners[4];
  Eigen::Vector3d ir_markers[4];
};

Eigen::Vector3d initial_position;
Eigen::Quaterniond initial_orientation;
vector<Eigen::Vector3d> waypoints_pos;
vector<Eigen::Vector3d> waypoints_vel;
vector<double> waypoints_yaw;

vector<Gate> gates;
int current_gate = -1;

ros::Publisher vis_pub_gates;
ros::Publisher vis_pub_wp;

ros::Publisher pub_gate_info;
ros::Publisher pub_pace_note;
ros::Publisher pub_waypoints;

void publish_gate_markers(void)
{
    visualization_msgs::Marker gates_marker;
    gates_marker.header.frame_id = "world";
    gates_marker.header.stamp = ros::Time();
    gates_marker.ns = "pace_notes";
    gates_marker.id = 0;
    gates_marker.type = visualization_msgs::Marker::LINE_LIST;
    gates_marker.action = visualization_msgs::Marker::ADD;
    gates_marker.pose.position.x = 0;
    gates_marker.pose.position.y = 0;
    gates_marker.pose.position.z = 0;
    gates_marker.pose.orientation.x = 0.0;
    gates_marker.pose.orientation.y = 0.0;
    gates_marker.pose.orientation.z = 0.0;
    gates_marker.pose.orientation.w = 1.0;
    gates_marker.scale.x = 0.1;
    gates_marker.color.a = 1.0;
    gates_marker.color.r = 0.0;
    gates_marker.color.g = 1.0;
    gates_marker.color.b = 0.0;
    for (int gate_idx = 0; gate_idx < gates.size(); gate_idx++) {
        for (int corner_idx = 0; corner_idx < 4; corner_idx++) {
            geometry_msgs::Point p1;
            p1.x = gates[gate_idx].corners[corner_idx].x();
            p1.y = gates[gate_idx].corners[corner_idx].y();
            p1.z = gates[gate_idx].corners[corner_idx].z();
            gates_marker.points.push_back(p1);

            geometry_msgs::Point p2;
            p2.x = gates[gate_idx].corners[(corner_idx+1)%4].x();
            p2.y = gates[gate_idx].corners[(corner_idx+1)%4].y();
            p2.z = gates[gate_idx].corners[(corner_idx+1)%4].z();
            gates_marker.points.push_back(p2);
        }
    }
    vis_pub_gates.publish(gates_marker);


    visualization_msgs::Marker waypoints_marker;
    waypoints_marker.header.frame_id = "world";
    waypoints_marker.header.stamp = ros::Time();
    waypoints_marker.ns = "pace_notes";
    waypoints_marker.id = 0;
    waypoints_marker.type = visualization_msgs::Marker::SPHERE_LIST;
    waypoints_marker.action = visualization_msgs::Marker::ADD;
    waypoints_marker.pose.position.x = 0;
    waypoints_marker.pose.position.y = 0;
    waypoints_marker.pose.position.z = 0;
    waypoints_marker.pose.orientation.x = 0.0;
    waypoints_marker.pose.orientation.y = 0.0;
    waypoints_marker.pose.orientation.z = 0.0;
    waypoints_marker.pose.orientation.w = 1.0;
    waypoints_marker.scale.x = 0.2;
    waypoints_marker.scale.y = 0.2;
    waypoints_marker.scale.z = 0.2;
    waypoints_marker.color.a = 1.0;
    waypoints_marker.color.r = 1.0;
    waypoints_marker.color.g = 0.0;
    waypoints_marker.color.b = 0.0;
    for (int wp_idx = 0; wp_idx < waypoints_pos.size(); wp_idx++) {
        geometry_msgs::Point p;
        p.x = waypoints_pos[wp_idx].x();
        p.y = waypoints_pos[wp_idx].y();
        p.z = waypoints_pos[wp_idx].z();
        waypoints_marker.points.push_back(p);
    }
    vis_pub_wp.publish(waypoints_marker);
}

void publish_pace_note(int gate_idx)
{
    vector<double> timestamps;
    vector<double> velocities;
    vector<long> measure_ir;

    if (gate_idx == 0) {
        timestamps.push_back(1.0);
        velocities.push_back(23.0);
        measure_ir.push_back(0);
        /*
        timestamps.push_back(3.0);
        velocities.push_back(20.0);
        measure_ir.push_back(0);
        */

    } else if (gate_idx == 1) {
        timestamps.push_back(1.0);
        velocities.push_back(23.0);
        measure_ir.push_back(0);
        timestamps.push_back(2.0);
        velocities.push_back(26.0);
        measure_ir.push_back(0);

        /*
        timestamps.push_back(2.2);
        velocities.push_back(20.0);
        measure_ir.push_back(1);
        */
        //timestamps.push_back(2.0);
        //velocities.push_back(20.0);
        //measure_ir.push_back(1);

    } else if (gate_idx == 2) {
        timestamps.push_back(0.0);
        velocities.push_back(24.0);
        measure_ir.push_back(0);
        timestamps.push_back(1.0);
        velocities.push_back(15.0);
        measure_ir.push_back(1);
        timestamps.push_back(1.5);
        velocities.push_back(15.0);
        measure_ir.push_back(1);

        //timestamps.push_back(3.0);
        //velocities.push_back(20.0);
        //measure_ir.push_back(0);

    } else if (gate_idx == 3) {
        timestamps.push_back(0.0);
        velocities.push_back(15.0);
        measure_ir.push_back(1);
        timestamps.push_back(1.0);
        velocities.push_back(15.0);
        measure_ir.push_back(1);
        timestamps.push_back(1.5);
        velocities.push_back(15.0);
        measure_ir.push_back(1);

    } else if (gate_idx == 4) {
        timestamps.push_back(0.0);
        velocities.push_back(20.0);
        measure_ir.push_back(0);
        timestamps.push_back(1.2);
        velocities.push_back(18.0);
        measure_ir.push_back(1);
        timestamps.push_back(1.8);
        velocities.push_back(20.0);
        measure_ir.push_back(1);

    } else if (gate_idx == 5) {
        timestamps.push_back(0.0);
        velocities.push_back(20.0);
        measure_ir.push_back(1);
        timestamps.push_back(0.5);
        velocities.push_back(19.0);
        measure_ir.push_back(1);


    } else if (gate_idx == 6) {
        timestamps.push_back(0.0);
        velocities.push_back(17.0);
        measure_ir.push_back(0);
        timestamps.push_back(1.0);
        velocities.push_back(18.0);
        measure_ir.push_back(0);


    } else if (gate_idx == 7) {
        timestamps.push_back(0.0);
        velocities.push_back(20.0);
        measure_ir.push_back(0);


    } else if (gate_idx == 8) {
        timestamps.push_back(0.0);
        velocities.push_back(19.0);
        measure_ir.push_back(0);
        timestamps.push_back(0.3);
        velocities.push_back(18.0);
        measure_ir.push_back(1);
        timestamps.push_back(0.8);
        velocities.push_back(20.0);
        measure_ir.push_back(1);
        timestamps.push_back(1.0);
        velocities.push_back(20.0);
        measure_ir.push_back(0);
        timestamps.push_back(1.2);
        velocities.push_back(20.0);
        measure_ir.push_back(1);

    } else if (gate_idx == 9) {
        timestamps.push_back(0.0);
        velocities.push_back(20.0);
        measure_ir.push_back(1);

        timestamps.push_back(1.0);
        velocities.push_back(20.0);
        measure_ir.push_back(1);

        timestamps.push_back(1.5);
        velocities.push_back(20.0);
        measure_ir.push_back(1);

        timestamps.push_back(2.0);
        velocities.push_back(20.0);
        measure_ir.push_back(1);

    } else if (gate_idx == 10) {
        timestamps.push_back(0.0);
        velocities.push_back(20.0);
        measure_ir.push_back(1);

        timestamps.push_back(0.5);
        velocities.push_back(20.0);
        measure_ir.push_back(1);

        timestamps.push_back(1.0);
        velocities.push_back(20.0);
        measure_ir.push_back(1);
    }
    
    puffin_pilot::PaceNote pn_msg;

    pn_msg.timestamps.layout.dim.push_back(std_msgs::MultiArrayDimension());
    pn_msg.timestamps.layout.dim[0].label  = "timestamps";
    pn_msg.timestamps.layout.dim[0].size   = timestamps.size();
    pn_msg.timestamps.layout.dim[0].stride = 1;
    pn_msg.timestamps.layout.data_offset   = 0;
    pn_msg.timestamps.data = timestamps;

    pn_msg.velocities.layout.dim.push_back(std_msgs::MultiArrayDimension());
    pn_msg.velocities.layout.dim[0].label  = "velocities";
    pn_msg.velocities.layout.dim[0].size   = velocities.size();
    pn_msg.velocities.layout.dim[0].stride = 1;
    pn_msg.velocities.layout.data_offset   = 0;
    pn_msg.velocities.data = velocities;

    pn_msg.measure_ir.layout.dim.push_back(std_msgs::MultiArrayDimension());
    pn_msg.measure_ir.layout.dim[0].label  = "measure_ir";
    pn_msg.measure_ir.layout.dim[0].size   = measure_ir.size();
    pn_msg.measure_ir.layout.dim[0].stride = 1;
    pn_msg.measure_ir.layout.data_offset   = 0;
    pn_msg.measure_ir.data = measure_ir;

    pub_pace_note.publish(pn_msg);
    ROS_INFO("Pace notes pace note #%d sent.", gate_idx);
}

void publish_gate_info(int gate_idx)
{
    std_msgs::Float64MultiArray ir_markers;
    ir_markers.layout.dim.push_back(std_msgs::MultiArrayDimension());
    ir_markers.layout.dim.push_back(std_msgs::MultiArrayDimension());
    ir_markers.layout.dim[0].label  = "ir_markers";
    ir_markers.layout.dim[0].size   = 4;
    ir_markers.layout.dim[0].stride = 4 * 3;
    ir_markers.layout.dim[1].label  = "coordinates";
    ir_markers.layout.dim[1].size   = 3;
    ir_markers.layout.dim[1].stride = 3;
    ir_markers.layout.data_offset   = 0;
    std::vector<double> gate_corners_data(4 * 3, 0);
    for (int corner_idx = 0; corner_idx < 4; corner_idx++) {
        gate_corners_data[corner_idx * 3 + 0] = gates[gate_idx].ir_markers[corner_idx].x();
        gate_corners_data[corner_idx * 3 + 1] = gates[gate_idx].ir_markers[corner_idx].y();
        gate_corners_data[corner_idx * 3 + 2] = gates[gate_idx].ir_markers[corner_idx].z();
    }
    ir_markers.data = gate_corners_data;

    puffin_pilot::GateInfo gate_info;
    //if (gate_idx == 7) {
    //    gate_info.gate_name.data = "nono";
    //} else {
        gate_info.gate_name.data = gates[gate_idx].name.c_str();
    //}
    gate_info.ir_markers = ir_markers;
    gate_info.header.stamp = ros::Time::now();
    gate_info.header.frame_id = "1";
    pub_gate_info.publish(gate_info);
}

void publish_waypoints(void)
{
    puffin_pilot::Waypoints wp_msg;

    wp_msg.positions.layout.dim.push_back(std_msgs::MultiArrayDimension());
    wp_msg.positions.layout.dim.push_back(std_msgs::MultiArrayDimension());
    wp_msg.positions.layout.dim[0].label  = "positions";
    wp_msg.positions.layout.dim[0].size   = waypoints_pos.size();
    wp_msg.positions.layout.dim[0].stride = waypoints_pos.size() * 3;
    wp_msg.positions.layout.dim[1].label  = "coordinates";
    wp_msg.positions.layout.dim[1].size   = 3;
    wp_msg.positions.layout.dim[1].stride = 3;
    wp_msg.positions.layout.data_offset   = 0;
    std::vector<double> positions_data(waypoints_pos.size() * 3, 0);
    for (int idx = 0; idx < waypoints_pos.size(); idx++) {
        positions_data[idx * 3 + 0] = waypoints_pos[idx].x();
        positions_data[idx * 3 + 1] = waypoints_pos[idx].y();
        positions_data[idx * 3 + 2] = waypoints_pos[idx].z();
    }
    wp_msg.positions.data = positions_data;

    wp_msg.velocities.layout.dim.push_back(std_msgs::MultiArrayDimension());
    wp_msg.velocities.layout.dim.push_back(std_msgs::MultiArrayDimension());
    wp_msg.velocities.layout.dim[0].label  = "velocities";
    wp_msg.velocities.layout.dim[0].size   = waypoints_vel.size();
    wp_msg.velocities.layout.dim[0].stride = waypoints_vel.size() * 3;
    wp_msg.velocities.layout.dim[1].label  = "coordinates";
    wp_msg.velocities.layout.dim[1].size   = 3;
    wp_msg.velocities.layout.dim[1].stride = 3;
    wp_msg.velocities.layout.data_offset   = 0;
    std::vector<double> velocities_data(waypoints_vel.size() * 3, 0);
    for (int idx = 0; idx < waypoints_vel.size(); idx++) {
        velocities_data[idx * 3 + 0] = waypoints_vel[idx].x();
        velocities_data[idx * 3 + 1] = waypoints_vel[idx].y();
        velocities_data[idx * 3 + 2] = waypoints_vel[idx].z();
    }
    wp_msg.velocities.data = velocities_data;

    wp_msg.yaws.layout.dim.push_back(std_msgs::MultiArrayDimension());
    wp_msg.yaws.layout.dim[0].label  = "yaws";
    wp_msg.yaws.layout.dim[0].size   = waypoints_yaw.size();
    wp_msg.yaws.layout.dim[0].stride = 1;
    std::vector<double> yaws_data(waypoints_yaw.size(), 0);
    for (int idx = 0; idx < waypoints_yaw.size(); idx++) {
        yaws_data[idx] = waypoints_yaw[idx];
    }
    wp_msg.yaws.data = yaws_data;

    pub_waypoints.publish(wp_msg);
    ROS_INFO("Pace notes waypoints sent.");
}

void append_waypoint(Eigen::Vector3d position, double yaw)
{
    waypoints_pos.push_back(position);
    waypoints_yaw.push_back(yaw);
    waypoints_vel.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
}

void init_gates(void)
{
    static const int corner_map[11][4] = {{1, 0, 2, 3},
                                          {0, 1, 2, 3},
                                          {1, 0, 3, 2},
                                          {1, 0, 3, 2},
                                          {0, 1, 3, 2},
                                          {1, 0, 3, 2},
                                          {0, 1, 2, 3},
                                          {0, 1, 2, 3},
                                          {0, 1, 2, 3},
                                          {1, 0, 2, 3},
                                          {1, 0, 2, 3}};


    

    std::vector<double> _initial_pose;
    ros::param::get("/uav/flightgoggles_uav_dynamics/init_pose", _initial_pose);
    initial_position.x() = _initial_pose.at(0);
    initial_position.y() = _initial_pose.at(1);
    initial_position.z() = _initial_pose.at(2);
    initial_orientation.x() = _initial_pose.at(3);
    initial_orientation.y() = _initial_pose.at(4);
    initial_orientation.z() = _initial_pose.at(5);
    initial_orientation.w() = _initial_pose.at(6);

    XmlRpc::XmlRpcValue gate_names;
    ros::param::get("/uav/gate_names", gate_names);
    gate_count = gate_names.size();

    append_waypoint(Eigen::Vector3d(18.0, -23.0, 5.3), 0.0);

    for (int gate_idx = 0; gate_idx < gate_count; gate_idx++) {
        const char *gate_name = static_cast<std::string>(gate_names[gate_idx]).c_str();
        char param_name[40];
        sprintf(param_name, "/uav/%s/nominal_location", gate_name);

        XmlRpc::XmlRpcValue nominal_location;
        ros::param::get(param_name, nominal_location);

        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        Eigen::Vector3d normal;
        
        Gate gate;
        gate.name = static_cast<std::string>(gate_names[gate_idx]).c_str();
        for (int corner_idx = 0; corner_idx < 4; corner_idx++) {
            gate.corners[corner_idx].x()    = (double)nominal_location[corner_map[gate_idx][corner_idx]][0];
            gate.corners[corner_idx].y()    = (double)nominal_location[corner_map[gate_idx][corner_idx]][1];
            gate.corners[corner_idx].z()    = (double)nominal_location[corner_map[gate_idx][corner_idx]][2];
            gate.ir_markers[corner_idx].x() = (double)nominal_location[corner_idx][0];
            gate.ir_markers[corner_idx].y() = (double)nominal_location[corner_idx][1];
            gate.ir_markers[corner_idx].z() = (double)nominal_location[corner_idx][2];
            center += gate.corners[corner_idx];
        }

        center /= 4;
        normal = (gate.corners[1] - gate.corners[0]).cross(gate.corners[2] - gate.corners[0]).normalized();
        append_waypoint(Eigen::Vector3d(0.0, 0.0, 0.0), 0.0); // Emtpy waypoint, modify later
        append_waypoint(center + normal * 0.0, 0.0);
        gate.normal = normal;

        printf("gate %d c(% 7.4f % 7.4f % 7.4f) n(% 7.4f % 7.4f % 7.4f)\n", 
                   gate_idx, 
                   center.x(), center.y(), center.z(),
                   normal.x(), normal.y(), normal.z());
        gates.push_back(gate);
    }

    for (int wp_idx = 1; wp_idx < waypoints_pos.size() - 1; wp_idx += 2) {
        waypoints_pos[wp_idx] = (waypoints_pos[wp_idx - 1] + waypoints_pos[wp_idx + 1]) / 2.0;
    }
    Eigen::Vector3d normal = (waypoints_pos[waypoints_pos.size()-1] - waypoints_pos[waypoints_pos.size()-2]).normalized();
    //waypoints_pos[waypoints_pos.size()-1] = waypoints_pos[waypoints_pos.size()-2] + normal * 10.0;
    append_waypoint(waypoints_pos[waypoints_pos.size()-1] + normal * 10.0, waypoints_yaw[waypoints_pos.size()-1]);
    append_waypoint(waypoints_pos[waypoints_pos.size()-1] + normal * 10.0, waypoints_yaw[waypoints_pos.size()-1]);
    append_waypoint(waypoints_pos[waypoints_pos.size()-1] + normal * 10.0, waypoints_yaw[waypoints_pos.size()-1]);

    for (int i = 0; i < 4; i++) {
        //append_waypoint(waypoints_pos[waypoints_pos.size()-1] + normal * 1.0, waypoints_yaw[waypoints_pos.size()-1]);
    }

    static const double waypoints_adjustment[26][3] = {{  0.0,   0.0,   0.0},
                                                       {  0.0,  -1.0,   0.2},
                                                       {  0.0,   0.0,   0.0}, // gate 0
                                                       {  0.6,   0.0,   0.0},
                                                       { -1.4,   1.0,   0.2}, // gate 1
                                                       { -6.5,   1.0,   0.0},
                                                       {  0.0,   0.0,   1.0}, // gate 2
                                                       {  0.0,   1.5,   0.5},
                                                       { -0.2,  -1.0,   0.2}, // gate 3
                                                       { -2.5,  -1.0,   2.5},
                                                       {  0.0,   0.0,   0.0}, // gate 4
                                                       { -1.0,  -0.5,   1.7},
                                                       {  0.0,   0.0,   1.0}, // gate 5
                                                       {  0.0,  -2.0,   1.2},
                                                       {  0.0,   0.0,   0.0}, // gate 6
                                                       {  0.0,  -2.0,   1.2},
                                                       {  0.0,  -1.0,   1.0}, // gate 7
                                                       {  0.0,  -2.0,   1.9},
                                                       { -0.8,   2.0,   1.0}, // gate 8
                                                       { -3.0,  -1.0,   2.5},
                                                       {  0.0,  -1.5,   0.0}, // gate 9
                                                       {  0.0,  -0.0,   1.0},
                                                       {  0.0,   0.0,   0.0}, // gate 10
                                                       {  0.0,   0.0,   0.0},
                                                       {  0.0,   0.0,   0.0},
                                                       {  0.0,   0.0,   0.0}};

    for (int i = 0; i < waypoints_pos.size(); i++) {
        waypoints_pos[i].x() += waypoints_adjustment[i][0];
        waypoints_pos[i].y() += waypoints_adjustment[i][1];
        waypoints_pos[i].z() += waypoints_adjustment[i][2];
    }

    waypoints_yaw[0] = 1.57;
    for (int wp_idx = 1; wp_idx < waypoints_pos.size() - 1; wp_idx++) {
        Eigen::Vector3d diff = waypoints_pos[wp_idx + 1] - waypoints_pos[wp_idx - 0];
        double next_yaw = atan2(diff.y(), diff.x());
        while (next_yaw + M_PI < waypoints_yaw[wp_idx - 1]) next_yaw += 2 * M_PI;
        while (next_yaw - M_PI > waypoints_yaw[wp_idx - 1]) next_yaw -= 2 * M_PI;
        waypoints_yaw[wp_idx] = next_yaw;

        printf("waypoint %2d (% 7.2f % 7.2f % 7.2f % 7.2f)\n",
               wp_idx,
               waypoints_pos[wp_idx].x(),
               waypoints_pos[wp_idx].y(),
               waypoints_pos[wp_idx].z(),
               waypoints_yaw[wp_idx]);
    }
    waypoints_yaw[waypoints_pos.size()-1] = 7.81;

    waypoints_yaw[12] += 0.4; // Between gate 5 and 6
    waypoints_yaw[13] += 0.4; // Between gate 5 and 6

    waypoints_yaw[17] += 0.4; // Between gate 7 and 8
    waypoints_yaw[17] += 0.4; // Between gate 7 and 8
    waypoints_yaw[18] += 0.3; // Between gate 7 and 8

    waypoints_yaw[21] -= 0.4; // Between gate 9 and 10
    waypoints_yaw[22] -= 0.1; // Between gate 9 and 10


    for (int wp_idx = 0; wp_idx < waypoints_vel.size() - 1; wp_idx++) {
        waypoints_vel[wp_idx] = (waypoints_pos[wp_idx + 1] - waypoints_pos[wp_idx]).normalized();
    }

    publish_gate_info(0);
}

void ir_marker_odometry_callback(const geometry_msgs::PoseStamped& msg)
{
    static int first = true;
    if (first) {
        Eigen::Vector3d new_position;
        Eigen::Quaterniond new_orientation;
        tf::pointMsgToEigen(msg.pose.position, new_position);
        tf::quaternionMsgToEigen(msg.pose.orientation, new_orientation);

        double x_diff = gates[0].corners[0].x();

        // Locate first gate before takeoff
        for (int idx = 0; idx < 4; idx++) {
            Eigen::Vector3d p;

            p = gates[0].corners[idx];
            p = p - new_position;
            p = new_orientation.inverse() * p;
            p = initial_orientation * p;
            p = p + initial_position;
            gates[0].corners[idx] = p;

            p = gates[0].ir_markers[idx];
            p = p - new_position;
            p = new_orientation.inverse() * p;
            p = initial_orientation * p;
            p = p + initial_position;
            gates[0].ir_markers[idx] = p;
        }

        waypoints_pos[2].x() += gates[0].corners[0].x() - x_diff;

        current_gate = 0;
        publish_gate_info(0);
        publish_gate_markers();
        publish_waypoints();
        publish_pace_note(0);

        first = false;
    }
}

double get_distance_from_gate(Eigen::Vector3d pos)
{
    Eigen::Vector3d center = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d normal = Eigen::Vector3d(0, 0, 0);

    for (int corner = 0; corner < 4; corner++) {
        center += gates[current_gate].corners[corner];
    }
    center /= 4;    
    normal = gates[current_gate].corners[1] - gates[current_gate].corners[0];
    normal = normal.cross(gates[current_gate].corners[2] - gates[current_gate].corners[0]);
    normal = normal.normalized();

    double distance = normal.dot(center - pos);
    return distance;
}

void odometry_callback(const nav_msgs::Odometry& msg)
{
    ROS_INFO_ONCE("Pace notes received first odometry message.");

    if (current_gate < 0) {
        return;
    }

    static const double gate_pass_distances[11] = {1.0,
                                                   1.0,
                                                   3.0,
                                                   1.0,
                                                   1.0,
                                                   2.0,
                                                   1.0,
                                                   1.0,
                                                   1.0,
                                                   1.0,
                                                   1.0};

    mav_msgs::EigenOdometry odometry;
    eigenOdometryFromMsg(msg, &odometry);

    if (get_distance_from_gate(odometry.position_W) < gate_pass_distances[current_gate]) {
        if (current_gate == gates.size()-1) {
            ROS_INFO_ONCE("Pace notes finished!");
        } else {
            current_gate += 1;
            publish_gate_info(current_gate);
            publish_pace_note(current_gate);
        }
    }
}

int main(int argc, char** argv)
{  
    printf("c\n");
    ros::init(argc, argv, "pace_notes");

    ros::NodeHandle nh;
    pub_gate_info = nh.advertise<puffin_pilot::GateInfo>("gate_info",  10, true);
    pub_pace_note = nh.advertise<puffin_pilot::PaceNote>("pace_note", 10, true);
    pub_waypoints = nh.advertise<puffin_pilot::Waypoints>("waypoints", 10, true);
    vis_pub_gates = nh.advertise<visualization_msgs::Marker>("/puff_pilot/gate_markers", 1, true);
    vis_pub_wp    = nh.advertise<visualization_msgs::Marker>("/puff_pilot/waypoint_markers", 1, true);

    printf("d\n");
    ros::Subscriber odometry_node        = nh.subscribe("odometry",        1, &odometry_callback,           ros::TransportHints().tcpNoDelay());
    ros::Subscriber irodom_callback_node = nh.subscribe("ir_markers_pose", 1, &ir_marker_odometry_callback, ros::TransportHints().tcpNoDelay());

    printf("e\n");
    init_gates();

    ros::spin();
    return 0;
}
