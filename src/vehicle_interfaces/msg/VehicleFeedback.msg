# VehicleFeedback.msg
# Feedback from Arduino (low-level) back to Pi (mid-level)
#
# header:          standard ROS2 header with timestamp
# actual_velocity: measured velocity in m/s (computed from encoder RPM)
# actual_rpm:      measured motor RPM from encoder
# encoder_ticks:   cumulative encoder tick count (for odometry)

std_msgs/Header header

float32 actual_velocity
float32 actual_rpm
int32 encoder_ticks
