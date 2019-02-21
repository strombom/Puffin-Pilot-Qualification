#include <ros/ros.h>

using namespace std;


int main(int argc, char** argv)
{
    ros::init(argc, argv, "puffin_state_estimation");

    ros::spin();
    return 0;
}
