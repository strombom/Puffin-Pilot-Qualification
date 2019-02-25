#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <iostream>

#include <puffin_pilot/PaceNote.h>

using namespace std;

int main(int argc, char** argv)
{  
    ros::init(argc, argv, "pace_notes");
    ros::NodeHandle n("~");

    ros::Publisher pub_pace_note = n.advertise<puffin_pilot::PaceNote>("pace_note", 10, true);

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

    pub_pace_note.publish(pace_note);

    ros::spinOnce();

    ros::waitForShutdown();
}
