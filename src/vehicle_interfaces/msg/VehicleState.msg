# VehicleState.msg
# Unified vehicle state used by closed-loop controllers
#
# header:          standard ROS2 header with timestamp and frame_id
# x:               position X in world frame [m]
# y:               position Y in world frame [m]
# yaw:             heading angle in world frame [rad]
# velocity:        longitudinal speed [m/s]
# yaw_rate:        angular velocity about Z [rad/s]
# steering_angle:  current front wheel steering angle [rad]

std_msgs/Header header

float32 x
float32 y
float32 yaw
float32 velocity
float32 yaw_rate
float32 steering_angle
