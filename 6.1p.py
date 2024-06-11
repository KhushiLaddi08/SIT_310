import sys
import time

# numpy
import numpy as np

# OpenCV
import cv2
from cv_bridge import CvBridge

# ROS Libraries
import rospy
import roslib

# ROS Message Types
from sensor_msgs.msg import CompressedImage

class Lane_Detector:
    def _init_(self):
        # Initialize the CvBridge class
        self.cv_bridge = CvBridge()

        # Subscribe to the image topic (remember to change the topic name if needed)
        self.image_sub = rospy.Subscriber('/akandb/camera_node/image/compressed', CompressedImage, self.image_callback, queue_size=1)

        # Initialize the ROS node
        rospy.init_node("my_lane_detector")

    def image_callback(self, msg):
        rospy.loginfo("image_callback")

        # Convert the compressed ROS image message to an OpenCV image
        img = self.cv_bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

        # Print some properties of the image
        print("Image is of type: ", type(img))
        print("Shape of image: ", img.shape)
        print("Size of image: ", img.size)
        print("Image stores elements of type: ", img.dtype)

        # Convert image to RGB space (OpenCV uses BGR by default)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect and process white lines
        lower_white = np.array([200, 200, 200])
        upper_white = np.array([255, 255, 255])
        mask_white = cv2.inRange(img_rgb, lower_white, upper_white)
        img_out_w = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_white)
        edges_white = cv2.Canny(img_out_w, 75, 150)
        lines_white = self.hough_transform(edges_white)

        # Detect and process yellow lines
        lower_yellow = np.array([100, 100, 0])
        upper_yellow = np.array([255, 255, 150])
        mask_yellow = cv2.inRange(img_rgb, lower_yellow, upper_yellow)
        img_out_y = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_yellow)
        edges_yellow = cv2.Canny(img_out_y, 75, 150)
        lines_yellow = self.hough_transform(edges_yellow)

        # Draw lines on the original image
        self.draw_lines(img, lines_white, (255, 0, 0))  # Red for white lines
        self.draw_lines(img, lines_yellow, (0, 255, 255))  # Yellow for yellow lines

        # Show the processed image
        cv2.imshow('Lane Lines', img)
        cv2.waitKey(1)

    def hough_transform(self, img):
        # Parameters for Hough Transform
        rho = 1
        theta = np.pi / 180
        threshold = 150
        min_line_length = 50
        max_line_gap = 20

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(img, rho, theta, threshold, min_line_length, max_line_gap)
        return lines

    def draw_lines(self, img, lines, color):
        # Draw detected lines on the image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), color, 3)

    def run(self):
        rospy.spin()  # Spin forever but listen to message callbacks

if _name_ == "_main_":
    try:
        lane_detector_instance = Lane_Detector()
        lane_detector_instance.run()
    except rospy.ROSInterruptException:
        pass
