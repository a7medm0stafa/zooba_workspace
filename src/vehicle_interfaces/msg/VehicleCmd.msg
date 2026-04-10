# VehicleCmd.msg
# Command message from mid-level controller to low-level controller
#
# velocity: target velocity in m/s
#           positive = forward, negative = reverse
# heading:  steering angle in degrees from center
#           positive = right, negative = left

float32 velocity
float32 heading
