# VehicleCmd.msg
# Command message from mid-level controller to low-level controller
#
# header:   standard ROS2 header with timestamp and frame_id
# velocity: target velocity in m/s
#           positive = forward, negative = reverse
# heading:  steering angle in degrees from center
#           positive = left (CCW), negative = right (CW)  [REP-103]

std_msgs/Header header

float32 velocity
float32 heading
