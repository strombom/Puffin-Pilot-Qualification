#include <ros/ros.h>
#include <iostream>
//#include <RateThrustControllerConfig.h>

#include <dynamic_reconfigure/server.h>
#include <rate_thrust_controller/RateThrustControllerConfig.h>


using namespace std;


void controller_dyn_config_callback(rate_thrust_controller::RateThrustControllerConfig &config, uint32_t level)
{
    ROS_INFO("Reconfigure request : % 7.2f % 7.2f % 7.2f % 7.2f % 7.2f % 7.2f ",
                config.p_gain_roll,
                config.p_gain_pitch,
                config.i_gain_roll,
                config.i_gain_pitch,
                config.d_gain_roll,
                config.d_gain_pitch);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "rate_thrust_controller");
    ros::NodeHandle node_handle("~");

    // Dynamic configuration
    dynamic_reconfigure::Server<rate_thrust_controller::RateThrustControllerConfig> controller_dyn_config_server_;
    controller_dyn_config_server_.setCallback(&controller_dyn_config_callback);

    ros::Rate loop_rate(10);
    while (ros::ok()) {

        ros::spinOnce();
    }
}


/*

void PIDAttitudeController::ComputeDesiredAngularVel(Eigen::Vector3d* angular_vel)
{
  assert(angular_vel);

  Eigen::Vector3d current_rpy;
  odometry_.getEulerAngles(&current_rpy);

  // önskad vinkel - nuvarande vinkel
  double roll_error = attitude_thrust_reference_(0) - current_rpy(0);
  double error_pitch = attitude_thrust_reference_(1) - current_rpy(1);

  // I
  roll_error_integration_  += roll_error;
  pitch_error_integration_ += error_pitch;

  // Tröskelvärde I
  if (std::abs(roll_error_integration_) > max_integrator_error_)
    roll_error_integration_ = max_integrator_error_ * roll_error_integration_
        / std::abs(roll_error_integration_);

  if (std::abs(pitch_error_integration_) > max_integrator_error_)
    pitch_error_integration_ = max_integrator_error_ * pitch_error_integration_
        / std::abs(pitch_error_integration_);


  //desired omega = [0;0;0]
  //double error_p = 0 - odometry_.angular_velocity_B(0);
  //double error_q = 0 - odometry_.angular_velocity_B(1);

  static double previous_error_roll = 0;

  double roll_error_derivative = roll_error - previous_error_roll;
  
  previous_error_roll = roll_error;


  roll_error
  roll_error_integration
  roll_error_derivative

  *angular_vel <<
     roll_error  * gain_p + roll_error_integration  * gain_i + roll_error_derivative  * gain_d,
     pitch_error * gain_p + pitch_error_integration * gain_i + pitch_error_derivative * gain_d,
     attitude_thrust_reference_(2);



  roll_gain_ * error_roll + // Vinkelfelet
  p_gain_ * error_p + 
  roll_integrator_gain_ * roll_error_integration_,





  *angular_vel
      <<  roll_gain_ * error_roll + p_gain_ * error_p + roll_integrator_gain_ * roll_error_integration_, 
          pitch_gain_ * error_pitch + q_gain_ * error_q + pitch_integrator_gain_ * pitch_error_integration_,
          odometry_.angular_velocity_B(2);



*/
