from typing import Tuple

import cv2
import numpy as np

from .common import SoccerFieldColors
from .config import DeWarpConfig


class Draw_SoccerField:
    def __init__(self, config: DeWarpConfig):
        self.config: DeWarpConfig = config

        # Create a blank image (dark green background)
        self.field_image = np.full(
            (int(config.field_width_total * config.ppm), int(config.field_length_total * config.ppm), 3),
            SoccerFieldColors.DarkGreen.value,
            dtype=np.uint8,
        )

    def draw_line(self, start_point: Tuple[float, float], end_point: Tuple[float, float]):
        thickness = int(self.config.line_width * self.config.ppm)
        start_pixel = (int(start_point[0] * self.config.ppm), int(start_point[1] * self.config.ppm))
        end_pixel = (int(end_point[0] * self.config.ppm), int(end_point[1] * self.config.ppm))
        cv2.line(self.field_image, start_pixel, end_pixel, (SoccerFieldColors.White.value), thickness)

    def draw_circle(self, center: Tuple[float, float], radius: float):
        center_pixel = (int(center[0] * self.config.ppm), int(center[1] * self.config.ppm))
        cv2.circle(
            self.field_image,
            center_pixel,
            int(radius * self.config.ppm),
            (SoccerFieldColors.White.value),
            int(self.config.line_width * self.config.ppm),
        )

    def draw_spot(self, center: Tuple[float, float], radius: float):
        center_pixel = (int(center[0] * self.config.ppm), int(center[1] * self.config.ppm))
        cv2.circle(
            self.field_image,
            center_pixel,
            int(radius * self.config.ppm),
            (SoccerFieldColors.White.value),
            int(self.config.line_width * self.config.ppm),
        )

    def draw_rectangle(self, top_left: Tuple[float, float], bottom_right: Tuple[float, float]):
        top_left_pixel = (int(top_left[0] * self.config.ppm), int(top_left[1] * self.config.ppm))
        bottom_right_pixel = (int(bottom_right[0] * self.config.ppm), int(bottom_right[1] * self.config.ppm))
        cv2.rectangle(
            self.field_image,
            top_left_pixel,
            bottom_right_pixel,
            (SoccerFieldColors.White.value),
            int(self.config.line_width * self.config.ppm),
        )

    def draw_goal(self, center: Tuple[float, float], width: float, depth: float):
        # Goals are drawn as rectangles perpendicular to the field's length
        half_width = width / 2
        top_left = (center[0] - depth, center[1] - half_width)
        bottom_right = (center[0], center[1] + half_width)
        self.draw_rectangle(top_left, bottom_right)

    def generate_field(self):
        # Draw the safety zone
        self.draw_rectangle(
            (self.config.safe_zone, self.config.safe_zone),
            (
                self.config.field_length_total - self.config.safe_zone,
                self.config.field_width_total - self.config.safe_zone,
            ),
        )

        # Offset all field elements by the safety zone
        offset = self.config.safe_zone

        # Drawing the outline of the field
        self.draw_rectangle(
            (offset, offset), (self.config.field_length_total - offset, self.config.field_width_total - offset)
        )

        # Drawing the center line
        self.draw_line(
            (self.config.field_length_total / 2, offset),
            (self.config.field_length_total / 2, self.config.field_width_total - offset),
        )

        # Drawing the center circle
        self.draw_circle(
            (self.config.field_length_total / 2, self.config.field_width_total / 2), self.config.center_circle_radius
        )

        # Drawing the penalty areas
        # Only the x-coordinate (left and right positions) is adjusted by the offset
        self.draw_rectangle(
            (offset, (self.config.field_width_total - self.config.penalty_area_width) / 2),
            (
                self.config.penalty_area_length + offset,
                (self.config.field_width_total + self.config.penalty_area_width) / 2,
            ),
        )
        self.draw_rectangle(
            (
                self.config.field_length_total - self.config.penalty_area_length - offset,
                (self.config.field_width_total - self.config.penalty_area_width) / 2,
            ),
            (
                self.config.field_length_total - offset,
                (self.config.field_width_total + self.config.penalty_area_width) / 2,
            ),
        )

        # Drawing the goal areas
        # Only the x-coordinate (left and right positions) is adjusted by the offset
        self.draw_rectangle(
            (offset, (self.config.field_width_total - self.config.goal_area_width) / 2),
            (self.config.goal_area_length + offset, (self.config.field_width_total + self.config.goal_area_width) / 2),
        )
        self.draw_rectangle(
            (
                self.config.field_length_total - self.config.goal_area_length - offset,
                (self.config.field_width_total - self.config.goal_area_width) / 2,
            ),
            (
                self.config.field_length_total - offset,
                (self.config.field_width_total + self.config.goal_area_width) / 2,
            ),
        )

        # Drawing the goals
        # The goals are drawn at the start and end of the field, adjusted by the offset
        self.draw_goal((0 + offset, self.config.field_width_total / 2), self.config.goal_width, self.config.goal_depth)
        self.draw_goal(
            (self.config.field_length_total - self.config.goal_depth, self.config.field_width_total / 2),
            self.config.goal_width,
            self.config.goal_depth,
        )

        # Drawing Center Spot 0,0 FCS
        self.draw_spot((self.config.field_length_total / 2, self.config.field_width_total / 2), self.config.spot_radius)

        return self.field_image
