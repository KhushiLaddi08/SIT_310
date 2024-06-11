#!/usr/bin/env python3

# Import necessary libraries
import sys
import time
import numpy as np
import cv2
from cv_bridge import CvBridge
import rospy
import roslib
from sensor_msgs.msg import CompressedImage

class LaneDetector:
    def __init__(self):
        self.cv_bridge = CvBridge()

        # Initialize the ROS node
        rospy.init_node("lane_detector_node")

        # Subscribe to the image topic
        self.image_sub = rospy.Subscriber('/duckie/camera_node/image/compressed', CompressedImage, self.image_callback, queue_size=1)

    def image_callback(self, msg):
        rospy.loginfo("Image received")

        # Convert the compressed image message to an OpenCV image
        img = self.cv_bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

        # Define cropping parameters (adjust these as needed)
        top, bottom, left, right = 200, 400, 100, 500

        # Crop the image
        cropped_img = img[top:bottom, left:right]

        # Convert the cropped image to HSV color space
        hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

        # Define HSV range for white and yellow colors
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([255, 50, 255])
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])

        # Create masks for white and yellow colors
        white_mask = cv2.inRange(hsv_img, lower_white, upper_white)
        yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

        # Apply Canny Edge Detection
        edges_white = cv2.Canny(white_mask, 50, 150)
        edges_yellow = cv2.Canny(yellow_mask, 50, 150)

        # Apply Hough Transform to detect lines
        white_lines = self.hough_transform(edges_white)
        yellow_lines = self.hough_transform(edges_yellow)

        # Draw the detected lines on the cropped image
        self.draw_lines(cropped_img, white_lines)
        self.draw_lines(cropped_img, yellow_lines)

        # Display the resulting images
        cv2.imshow('White Mask', white_mask)
        cv2.imshow('Yellow Mask', yellow_mask)
        cv2.waitKey(1)

    def hough_transform(self, edges):
        # Apply Hough Transform
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=50)
        return lines

    def draw_lines(self, img, lines):
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        lane_detector = LaneDetector()
        lane_detector.run()
    except rospy.ROSInterruptException:
        pass
