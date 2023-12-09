#!/usr/bin/env python3

import cv2
import numpy as np

# Field Size and other dimensions for MSL field 18 x 12 meter 
field_length = 18  # meters
field_width = 12  # meters
penalty_area_length = 2.25  # E, meters
penalty_area_width = 6.5  # C, meters
goal_area_length = 0.75  # F, meters
goal_area_width = 3.5  # D, meters
center_circle_radius = 2  # H, meters
goal_depth = 0.5  # Goal depth,
goal_width = 2.0  # Goal width 2m for this field -> 2.4m allowed?
line_width = 0.125  # K, meters
ppm = 100  # pixels per meter
safe_zone = 1  # Safety zone around the field, meters

# Color
# Define basic colors for Soccer Field (RGB)
black = (0, 0, 0)
white = (255, 255, 255)
gray = (128, 128, 128)
red = (255, 0, 0)
green = (0, 255, 0)
lightgreen = (144, 238, 144)
darkgreen = (0, 100, 0) 
blue = (0, 0, 255)
lightblue = (173, 216, 230)
pink = (255, 192, 203)
magenta = (255, 0, 255)

class SoccerField:
    def __init__(self):
        self.length = field_length + 2 * safe_zone  # Adding safety zone to the length
        self.width = field_width + 2 * safe_zone  # Adding safety zone to the width
        self.line_width = line_width
        self.center_circle_radius = center_circle_radius
        self.ppm = ppm
        self.safe_zone = safe_zone

        # Create a blank image (dark green background)
        self.field_image = np.full((int(self.width * self.ppm), int(self.length * self.ppm), 3), darkgreen, dtype=np.uint8)

    def draw_line(self, start_point, end_point):
        thickness = int(self.line_width * self.ppm)
        start_pixel = (int(start_point[0] * self.ppm), int(start_point[1] * self.ppm))
        end_pixel = (int(end_point[0] * self.ppm), int(end_point[1] * self.ppm))
        cv2.line(self.field_image, start_pixel, end_pixel, (white), thickness)

    def draw_circle(self, center, radius):
        center_pixel = (int(center[0] * self.ppm), int(center[1] * self.ppm))
        cv2.circle(self.field_image, center_pixel, int(radius * self.ppm), (white), int(self.line_width * self.ppm))

    def draw_rectangle(self, top_left, bottom_right):
        top_left_pixel = (int(top_left[0] * self.ppm), int(top_left[1] * self.ppm))
        bottom_right_pixel = (int(bottom_right[0] * self.ppm), int(bottom_right[1] * self.ppm))
        cv2.rectangle(self.field_image, top_left_pixel, bottom_right_pixel, (white), int(self.line_width * self.ppm))

    def draw_goal(self, center, width, depth):
        # Goals are drawn as rectangles perpendicular to the field's length
        half_width = width / 2
        top_left = (center[0] - depth, center[1] - half_width)
        bottom_right = (center[0], center[1] + half_width)
        self.draw_rectangle(top_left, bottom_right)

    def generate_field(self):
        # Draw the safety zone
        self.draw_rectangle((self.safe_zone, self.safe_zone), (self.length - self.safe_zone, self.width - self.safe_zone))

        # Offset all field elements by the safety zone
        offset = self.safe_zone

        # Drawing the outline of the field
        self.draw_rectangle((offset, offset), (self.length - offset, self.width - offset))

        # Drawing the center line
        self.draw_line((self.length / 2, offset), (self.length / 2, self.width - offset))

        # Drawing the center circle
        self.draw_circle((self.length / 2, self.width / 2), self.center_circle_radius)

        # Drawing the penalty areas
        # Only the x-coordinate (left and right positions) is adjusted by the offset
        self.draw_rectangle((offset, (self.width - penalty_area_width) / 2), (penalty_area_length + offset, (self.width + penalty_area_width) / 2))
        self.draw_rectangle((self.length - penalty_area_length - offset, (self.width - penalty_area_width) / 2), (self.length - offset, (self.width + penalty_area_width) / 2))

        # Drawing the goal areas
        # Only the x-coordinate (left and right positions) is adjusted by the offset
        self.draw_rectangle((offset, (self.width - goal_area_width) / 2), (goal_area_length + offset, (self.width + goal_area_width) / 2))
        self.draw_rectangle((self.length - goal_area_length - offset, (self.width - goal_area_width) / 2), (self.length - offset, (self.width + goal_area_width) / 2))

        # Drawing the goals
        # The goals are drawn at the start and end of the field, adjusted by the offset
        self.draw_goal((0 + offset, self.width / 2), goal_width, goal_depth)
        self.draw_goal((self.length - goal_depth, self.width / 2), goal_width, goal_depth)

    def display_field(self):
        cv2.imshow('Soccer Field', self.field_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Example usage
field = SoccerField()  # RoboCup MSL field dimensions in meters
field.generate_field()
field.display_field()
