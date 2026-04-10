# VehicleConstraints.msg
# Diagnostic message published by the non-holonomic constraints node
# Reports current constraint parameters and enforcement status
#
# max_velocity:            maximum allowed velocity [m/s]
# max_steering_angle_deg:  maximum allowed steering angle [degrees]
# min_turning_radius:      minimum turning radius from Ackermann geometry [m]
# wheelbase:               distance between front and rear axles [m]
# constraints_active:      whether constraint enforcement is enabled

float32 max_velocity
float32 max_steering_angle_deg
float32 min_turning_radius
float32 wheelbase
bool constraints_active
