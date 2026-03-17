import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial


class VehicleActuatorNode(Node):

    def __init__(self):
        super().__init__('vehicle_actuator_node')

        # Subscriber to vehicle command topic
        self.subscription = self.create_subscription(
            String,
            'vehicle/command',
            self.command_callback,
            10
        )

        # Serial connection to Arduino
        self.serial_port = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

        self.get_logger().info("Vehicle actuator node started")


    def command_callback(self, msg):

        command = msg.data

        self.get_logger().info(f"Received command: {command}")

        if command == "STOP":

            self.serial_port.write(b'R\n')
            self.get_logger().info("Sent RED signal to Arduino")


        elif command == "GO":

            self.serial_port.write(b'G\n')
            self.get_logger().info("Sent GREEN signal to Arduino")


        elif command == "SLOW":

            self.serial_port.write(b'Y\n')
            self.get_logger().info("Sent YELLOW signal to Arduino")


        else:

            self.serial_port.write(b'O\n')
            self.get_logger().info("No signal detected")


def main(args=None):

    rclpy.init(args=args)

    node = VehicleActuatorNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.serial_port.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()