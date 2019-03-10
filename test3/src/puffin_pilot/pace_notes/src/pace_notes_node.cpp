#include <ros/ros.h>
#include <iostream>
#include <Eigen/Geometry>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Float64MultiArray.h>

#include <puffin_pilot/PaceNote.h>

using namespace std;


int gate_count;

struct Gate {
  string name;
  Eigen::Vector3d normal;
  Eigen::Vector3d corners[4];
  Eigen::Vector3d ir_markers[4];
  Eigen::Vector3d waypoints[3];
  double waypoints_yaw[3];
};

vector<Gate> gates;
int current_gate = 0;

ros::Publisher vis_pub_gates;
ros::Publisher vis_pub_gate_centers;
ros::Publisher pub_pace_note;

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
        gate.waypoints[0] = center - normal * 2;
        //gate.waypoints[1] = center;
        gate.waypoints[1] = center + normal * 1;
        gate.normal = normal;

        printf("gate %d c(% 7.4f % 7.4f % 7.4f) n(% 7.4f % 7.4f % 7.4f)\n", 
                   gate_idx, 
                   center.x(), center.y(), center.z(),
                   normal.x(), normal.y(), normal.z());
        gates.push_back(gate);
    }

    for (int gate_idx = 0; gate_idx < gates.size() - 1; gate_idx++) {
        gates[gate_idx].waypoints[2] = (gates[gate_idx].waypoints[1] + gates[gate_idx+1].waypoints[0]) / 2.0;
    }

    gates[gates.size()-1].waypoints[2] = gates[gates.size()-1].waypoints[1] + 20.0 * gates[gates.size()-1].normal;
}

void marker_publisher(const ros::TimerEvent&)
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


    visualization_msgs::Marker gate_centers;
    gate_centers.header.frame_id = "world";
    gate_centers.header.stamp = ros::Time();
    gate_centers.ns = "pace_notes";
    gate_centers.type = visualization_msgs::Marker::POINTS;
    gate_centers.action = visualization_msgs::Marker::ADD;
    gate_centers.pose.position.x = 0;
    gate_centers.pose.position.y = 0;
    gate_centers.pose.position.z = 0;
    gate_centers.pose.orientation.x = 0.0;
    gate_centers.pose.orientation.y = 0.0;
    gate_centers.pose.orientation.z = 0.0;
    gate_centers.pose.orientation.w = 1.0;
    gate_centers.scale.x = 0.2;
    gate_centers.scale.y = 0.2;
    gate_centers.scale.z = 0.2;
    gate_centers.color.a = 1.0;

    for (int point_idx = 0; point_idx < 3; point_idx++) {
        gate_centers.points.clear();
        for (int gate_idx = 0; gate_idx < gates.size(); gate_idx++) {
            geometry_msgs::Point p1;
            p1.x = gates[gate_idx].waypoints[point_idx].x();
            p1.y = gates[gate_idx].waypoints[point_idx].y();
            p1.z = gates[gate_idx].waypoints[point_idx].z();
            gate_centers.points.push_back(p1);
        }

        gate_centers.id = point_idx;
        gate_centers.color.r = 0.0;
        gate_centers.color.g = 0.0;
        gate_centers.color.b = 0.0;
        if (point_idx == 0) gate_centers.color.r = 1.0;
        if (point_idx == 1) gate_centers.color.g = 1.0;
        if (point_idx == 2) gate_centers.color.b = 1.0;
        if (point_idx == 3) {
            gate_centers.color.r = 1.0;
            gate_centers.color.g = 1.0;
        }

        vis_pub_gate_centers.publish(gate_centers);
    }


    std_msgs::Float64MultiArray ir_markers;
    ir_markers.layout.dim.push_back(std_msgs::MultiArrayDimension());
    ir_markers.layout.dim.push_back(std_msgs::MultiArrayDimension());
    ir_markers.layout.dim[0].label = "ir_markers";
    ir_markers.layout.dim[0].size = 4;
    ir_markers.layout.dim[0].stride = 4 * 3;
    ir_markers.layout.dim[1].label = "coordinates";
    ir_markers.layout.dim[1].size = 3;
    ir_markers.layout.dim[1].stride = 3;
    ir_markers.layout.data_offset = 0;
    std::vector<double> gate_corners_data(4 * 3, 0);
    for (int corner_idx = 0; corner_idx < 4; corner_idx++) {
        gate_corners_data[corner_idx * 3 + 0] = gates[current_gate].ir_markers[corner_idx].x();
        gate_corners_data[corner_idx * 3 + 1] = gates[current_gate].ir_markers[corner_idx].y();
        gate_corners_data[corner_idx * 3 + 2] = gates[current_gate].ir_markers[corner_idx].z();
    }
    ir_markers.data = gate_corners_data;


    std_msgs::Float64MultiArray waypoints;
    waypoints.layout.dim.push_back(std_msgs::MultiArrayDimension());
    waypoints.layout.dim.push_back(std_msgs::MultiArrayDimension());
    waypoints.layout.dim[0].label = "waypoints";
    waypoints.layout.dim[0].size = 3;
    waypoints.layout.dim[0].stride = 3 * 3;
    waypoints.layout.dim[1].label = "coordinates";
    waypoints.layout.dim[1].size = 3;
    waypoints.layout.dim[1].stride = 3;
    waypoints.layout.data_offset = 0;
    std::vector<double> waypoints_data(3 * 3, 0);
    for (int waypoint_idx = 0; waypoint_idx < 3; waypoint_idx++) {
        waypoints_data[waypoint_idx * 3 + 0] = gates[current_gate].waypoints[waypoint_idx].x();
        waypoints_data[waypoint_idx * 3 + 1] = gates[current_gate].waypoints[waypoint_idx].y();
        waypoints_data[waypoint_idx * 3 + 2] = gates[current_gate].waypoints[waypoint_idx].z();
    }
    waypoints.data = waypoints_data;





    puffin_pilot::PaceNote pace_note;
    pace_note.gate_name.data = gates[current_gate].name.c_str();
    pace_note.waypoints = waypoints;
    pace_note.ir_markers = ir_markers;
    pace_note.header.stamp = ros::Time::now();
    pace_note.header.frame_id = "1";
    pub_pace_note.publish(pace_note);
    ROS_INFO("Pace notes sent new.");




/*
    for (int corner_idx = 0; corner_idx < 4; corner_idx++) {

        visualization_msgs::Marker corner_marker;
        corner_marker.header.frame_id = "world";
        corner_marker.header.stamp = ros::Time();
        corner_marker.ns = "pace_notes";
        corner_marker.id = corner_idx + 1;
        corner_marker.type = visualization_msgs::Marker::SPHERE_LIST;
        corner_marker.action = visualization_msgs::Marker::ADD;
        corner_marker.pose.position.x = 0;
        corner_marker.pose.position.y = 0;
        corner_marker.pose.position.z = 0;
        corner_marker.pose.orientation.x = 0.0;
        corner_marker.pose.orientation.y = 0.0;
        corner_marker.pose.orientation.z = 0.0;
        corner_marker.pose.orientation.w = 1.0;
        corner_marker.scale.x = 0.4;
        corner_marker.scale.y = 0.4;
        corner_marker.scale.z = 0.4;

        corner_marker.color.a = 1.0;
        corner_marker.color.r = 0.0;
        corner_marker.color.g = 0.0;
        corner_marker.color.b = 0.0;


        for (int gate_idx = 0; gate_idx < gates.size(); gate_idx++) {
            geometry_msgs::Point p;
            p.x = gates[gate_idx].corners[corner_idx].x();
            p.y = gates[gate_idx].corners[corner_idx].y();
            p.z = gates[gate_idx].corners[corner_idx].z();
            corner_marker.points.push_back(p);
        }

        if (corner_idx == 0) {
            corner_marker.color.r = 1.0;
            vis_pub_1.publish(corner_marker);

        } else if (corner_idx == 1) {
            corner_marker.color.g = 1.0;
            vis_pub_2.publish(corner_marker);

        } else if (corner_idx == 2) {
            corner_marker.color.b = 1.0;
            vis_pub_3.publish(corner_marker);

        } else if (corner_idx == 3) {
            corner_marker.color.r = 1.0;
            corner_marker.color.g = 1.0;
            vis_pub_4.publish(corner_marker);

        }


    }
*/

}

int main(int argc, char** argv)
{  
    ros::init(argc, argv, "pace_notes");
    ros::NodeHandle nh;

    pub_pace_note = nh.advertise<puffin_pilot::PaceNote>("pace_note", 10, true);
    vis_pub_gates = nh.advertise<visualization_msgs::Marker>("pace_notes/markers/gates", 0);
    vis_pub_gate_centers = nh.advertise<visualization_msgs::Marker>("pace_notes/markers/gate_centers", 0);
    ros::Timer marker_publisher_timer = nh.createTimer(ros::Duration(2.0), marker_publisher);

    /*vis_pub_1 = nh.advertise<visualization_msgs::Marker>("pace_notes_markers1", 0);
    vis_pub_2 = nh.advertise<visualization_msgs::Marker>("pace_notes_markers2", 0);
    vis_pub_3 = nh.advertise<visualization_msgs::Marker>("pace_notes_markers3", 0);
    vis_pub_4 = nh.advertise<visualization_msgs::Marker>("pace_notes_markers4", 0);*/

    init_gates();

    ros::spin();
    return 0;

/*

    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time();
    marker.ns = "pace_notes";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = 16;
    marker.pose.position.y = 1;
    marker.pose.position.z = 5;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;

    ros::Rate loop_rate(1);
    while (ros::ok()) {
        vis_pub.publish( marker );
        ros::spinOnce();
        loop_rate.sleep();
    }




    puffin_pilot::PaceNote pace_note;
    std::stringstream ss;
    ss << "Gate10";
    pace_note.gate_name.data = ss.str();

    std_msgs::Float64MultiArray gate;
    //gate.layout.dim = (std_msgs::MultiArrayLayout *) malloc(sizeof(std_msgs::MultiArrayLayout) * 2);
    gate.layout.dim.push_back(std_msgs::MultiArrayDimension());
    gate.layout.dim.push_back(std_msgs::MultiArrayDimension());
    gate.layout.dim[0].label = "corners";
    gate.layout.dim[0].size = 4;
    gate.layout.dim[0].stride = 4 * 3;
    gate.layout.dim[1].label = "coordinates";
    gate.layout.dim[1].size = 3;
    gate.layout.dim[1].stride = 3;
    gate.layout.data_offset = 0;

    std::vector<double> gate_corners_data(4 * 3, 0);

    gate_corners_data[ 0] = 18.15111;
    gate_corners_data[ 1] = 3.631447;
    gate_corners_data[ 2] = 7.229498;
    
    gate_corners_data[ 3] = 16.35111;
    gate_corners_data[ 4] = 3.631447;
    gate_corners_data[ 5] = 7.229498;
    
    gate_corners_data[ 6] = 18.15111;
    gate_corners_data[ 7] = 3.631447;
    gate_corners_data[ 8] = 5.383497;
    
    gate_corners_data[ 9] = 16.35111;
    gate_corners_data[10] = 3.631447;
    gate_corners_data[11] = 5.383497;

    gate.data = gate_corners_data;
    pace_note.gate_corners = gate;

    pace_note.header.stamp = ros::Time::now();
    pace_note.header.frame_id = "1";


    //ros::Rate loop_rate(5);
    while (ros::ok()) {
        pub_pace_note.publish(pace_note);
        ros::spinOnce();
        loop_rate.sleep();
    }
*/

}
