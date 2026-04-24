# ImuData.msg
# Raw IMU data from the HW-123 (MPU6050) sensor via Arduino
#
# header:   standard ROS2 header with timestamp
# accel_x/y/z:  accelerometer readings [m/s²]
# gyro_x/y/z:   gyroscope readings [rad/s]
# yaw:          complementary-filter fused yaw from Arduino [degrees]

std_msgs/Header header

float32 accel_x
float32 accel_y
float32 accel_z
float32 gyro_x
float32 gyro_y
float32 gyro_z
float32 yaw
