#include <ros/ros.h>
#include <cmath>
#include <iostream>
#include <Eigen/Geometry>
#include <visualization_msgs/Marker.h>
#include <nav_msgs/Odometry.h>
#include <mav_msgs/conversions.h>
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

void publish_pace_note(void)
{
    vector<double> timestamps;
    vector<double> velocities;
    vector<long> measure_ir;

    if (current_gate == 0) {
        timestamps.push_back(2.0);
        velocities.push_back(-1.0);
        measure_ir.push_back(1);
        //timestamps.push_back(3.2);
        //velocities.push_back(0.0);
    } else {
        timestamps.push_back(2.0);
        velocities.push_back(-1.0);
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
    ROS_INFO("Pace notes pace note #%d sent.", current_gate);
}

void publish_gate_info(void)
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
        gate_corners_data[corner_idx * 3 + 0] = gates[current_gate].ir_markers[corner_idx].x();
        gate_corners_data[corner_idx * 3 + 1] = gates[current_gate].ir_markers[corner_idx].y();
        gate_corners_data[corner_idx * 3 + 2] = gates[current_gate].ir_markers[corner_idx].z();
    }
    ir_markers.data = gate_corners_data;

    puffin_pilot::GateInfo gate_info;
    gate_info.gate_name.data = gates[current_gate].name.c_str();
    gate_info.ir_markers = ir_markers;
    gate_info.header.stamp = ros::Time::now();
    gate_info.header.frame_id = "1";
    pub_gate_info.publish(gate_info);
    ROS_INFO("Pace notes gate info sent %d.", current_gate);

    publish_pace_note();
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

    XmlRpc::XmlRpcValue gate_names;
    ros::param::get("/uav/gate_names", gate_names);
    gate_count = gate_names.size();

    //append_waypoint(Eigen::Vector3d(18.0, -23.0, 5.3), 0.0);
    //append_waypoint(Eigen::Vector3d(18.0, -16.0, 5.8), 0.0);
    append_waypoint(Eigen::Vector3d(18.0, -23.0, 5.3), 0.0);
    //append_waypoint(Eigen::Vector3d(18.0, -16.0, 6.3), 0.0);

    for (int gate_idx = 0; gate_idx < gate_count; gate_idx++) {
        const char *gate_name = static_cast<std::string>(gate_names[gate_idx]).c_str();
        char param_name[40];
        sprintf(param_name, "/uav/%s/nominal_location", gate_name);

        XmlRpc::XmlRpcValue nominal_location;
        ros::param::get(param_name, nominal_location);

        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        Eigen::Vector3d normal;
        
        Gate gate;
        gate.name = static_cast<std::string>(gate_names[gate_idx]);
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
                                                       {  0.0,   0.0,   0.2},
                                                       {  0.0,   0.0,   0.0}, // gate 1
                                                       {  0.0,  10.0,   0.2},
                                                       { -1.5,   1.0,   0.2}, // gate 2
                                                       { -6.5,   0.0,   0.2},
                                                       {  0.0,   0.0,   0.2}, // gate 3
                                                       {  0.0,   0.0,   0.2},
                                                       {  0.5,   0.0,   0.2}, // gate 4
                                                       { -3.0,  -3.0,   1.0},
                                                       {  0.7,   0.0,   0.0}, // gate 5
                                                       { -1.0,   0.0,   0.7},
                                                       {  0.0,   0.0,   0.5}, // gate 6
                                                       {  0.0,  -1.5,   0.2},
                                                       {  0.0,  -1.2,   0.0}, // gate 7
                                                       {  0.0,  -1.5,   0.2},
                                                       {  0.0,  -1.0,   0.5}, // gate 8
                                                       {  1.0,   0.0,   0.7},
                                                       {  0.0,   2.0,   0.2}, // gate 9
                                                       { -3.0,  -1.0,   0.7},
                                                       { -0.7,  -1.5,   0.0}, // gate 10
                                                       {  0.0,  -5.0,   0.5},
                                                       {  0.0,   0.0,   0.0}, // gate 11
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

    for (int wp_idx = 0; wp_idx < waypoints_vel.size() - 1; wp_idx++) {
        waypoints_vel[wp_idx] = (waypoints_pos[wp_idx + 1] - waypoints_pos[wp_idx]).normalized();
    }

    publish_gate_markers();
    publish_waypoints();

    current_gate = 0;
    publish_gate_info();
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
                                                   1.0,
                                                   1.0,
                                                   1.0,
                                                   1.5,
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
            publish_gate_info();
        }
    }
}

int main(int argc, char** argv)
{  
    ros::init(argc, argv, "pace_notes");

    ros::NodeHandle nh;
    pub_gate_info = nh.advertise<puffin_pilot::GateInfo>("gate_info",  10, true);
    pub_pace_note = nh.advertise<puffin_pilot::PaceNote>("pace_note", 10, true);
    pub_waypoints = nh.advertise<puffin_pilot::Waypoints>("waypoints", 10, true);
    vis_pub_gates = nh.advertise<visualization_msgs::Marker>("/puff_pilot/gate_markers", 1, true);
    vis_pub_wp    = nh.advertise<visualization_msgs::Marker>("/puff_pilot/waypoint_markers", 1, true);

    ros::Subscriber odometry_node = nh.subscribe("odometry", 1,  &odometry_callback, ros::TransportHints().tcpNoDelay());

    init_gates();

    ros::spin();
    return 0;
}
