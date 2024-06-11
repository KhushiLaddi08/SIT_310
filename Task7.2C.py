r/bin/env python3

import rospy
from duckietown_msgs.msg import Twist2DStamped
from duckietown_msgs.msg import AprilTagDetectionArray

class Target_Follower:
    def _init_(self):
        # Initialize ROS node
        rospy.init_node('target_follower_node', anonymous=True)

        # When shutdown signal is received, we run clean_shutdown function
        rospy.on_shutdown(self.clean_shutdown)

        # Initialize publisher and subscriber with your robot's name
        self.cmd_vel_pub = rospy.Publisher('/shravel/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=1)
        rospy.Subscriber('/shravel/apriltag_detector_node/detections', AprilTagDetectionArray, self.tag_callback, queue_size=1)
        rospy.spin() # Spin forever but listen to message callbacks

    # April Tag Detection Callback
    def tag_callback(self, msg):
        self.move_robot(msg.detections)
 
    # Stop Robot before node has shut down. This ensures the robot keeps moving
with the last velocity command
        rospy.loginfo("System shutting down. Stopping robot...")
        self.stop_robot()

    # Sends zero velocity to stop the robot
    def stop_robot(self):
        cmd_msg = Twist2DStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.v = 0.0
        cmd_msg.omega = 0.0
        self.cmd_vel_pub.publish(cmd_msg)

    def move_robot(self, detections):
        if len(detections) == 2:
         x1 = detections[0].transform.translation.x
         y1 = detections[0].transform.translation.y
         z1 = detections[0].transform.translation.z
         rospy.loginfo("Sign 1 - x, y, z: %f, %f, %f", x1, y1, z1)
         x2 = detections[1].transform.translation.x
         y2 = detections[1].transform.translation.y
         z2 = detections[1].transform.translation.z
         rospy.loginfo("Sign 2 - x, y, z: %f, %f, %f", x2, y2, z2)
         rospy.sleep(1)
         if (z2 > z1):
          x = detections[0].transform.translation.x
          z = detections[0].transform.translation.z
         else:
          x = detections[1].transform.translation.x
          z = detections[1].transform.translation.z

         if z > 0.15:
          cmd_msg = Twist2DStamped()
          cmd_msg.header.stamp = rospy.Time.now()
          cmd_msg.v = 0.2
          cmd_msg.omega = 0
          self.cmd_vel_pub.publish(cmd_msg)
          rospy.sleep(0.2)
          self.stop_robot()
         if z < 0.10:
          cmd_msg = Twist2DStamped()
          cmd_msg.header.stamp = rospy.Time.now()
          cmd_msg.v = -0.2
          cmd_msg.omega = 0
          self.cmd_vel_pub.publish(cmd_msg)
          rospy.sleep(0.2)
          self.stop_robot()
         if x > 0.05:
          cmd_msg = Twist2DStamped()
          cmd_msg.header.stamp = rospy.Time.now()
          cmd_msg.v = 0
          cmd_msg.omega = -0.2
          self.cmd_vel_pub.publish(cmd_msg)
          rospy.sleep(0.2)
          self.stop_robot()
         if x > 0.05:
          cmd_msg = Twist2DStamped()
          cmd_msg.header.stamp = rospy.Time.now()
          cmd_msg.v = 0
          cmd_msg.omega = -0.2
          self.cmd_vel_pub.publish(cmd_msg)
          rospy.sleep(0.2)
          self.stop_robot()
         if x < -0.05:
          cmd_msg = Twist2DStamped()
          cmd_msg.header.stamp = rospy.Time.now()
          cmd_msg.v = 0
          cmd_msg.omega = 0.2
          self.cmd_vel_pub.publish(cmd_msg)
          rospy.sleep(0.2)
          self.stop_robot()
        else:
         self.stop_robot()
if _name_ == '_main_':
    try:
        target_follower = Target_Follower()
    except rospy.ROSInterruptException:
        pass
