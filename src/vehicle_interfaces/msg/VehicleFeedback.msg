# VehicleFeedback.msg
# Feedback from Arduino (low-level) back to Pi (mid-level)
#
# actual_velocity: measured velocity in m/s (computed from encoder RPM)
# actual_rpm:      measured motor RPM from encoder
# encoder_ticks:   cumulative encoder tick count (for odometry)

float32 actual_velocity
float32 actual_rpm
int32 encoder_ticks
