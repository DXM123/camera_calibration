from typing import Union
#import logging

import cv2
import numpy as np

from .common import MarkerColors, SoccerFieldColors
from .config import get_config

class Warper(object):
    def __init__(
        self,
        points: np.ndarray,
        landmark_points: np.ndarray,
        width: Union[int, float] = 640,
        height: Union[int, float] = 480,
        interpolation=None,
    ):
        self.config = get_config()

        # TODO The types of width and height were inconsistent with their default
        # I fixed them now to be `Union[int, float]`. Do they have to be dynamic
        # in case of resizing of frame? -> Yes, just initial staring values

        assert isinstance(width, (int, float)) or not isinstance(
            height, (int, float)
        ), "Width and height should be numerical values."
        
        #Give warper Class the attrbutes
        self.src_width = width
        self.src_height = height
        self.src_points = points
        self.landmark_points = landmark_points # Field Coordinate FCS
        self.dst = None

        #logging.info(
        print(
            "Following values used as input for Warper Class:"
            f" \nWidth: {self.src_width}.\nHeight: {self.src_height}.\n"
            f"Points: {self.src_points}.\nLandMarkPoints:{self.landmark_points}."
        )

        # Two way to calculate the Homography Hv (output is identical but one also produces mask (not used now))
        self.M = cv2.getPerspectiveTransform(self.src_points.astype(np.float32), self.landmark_points.astype(np.float32))
        #self.H, mask = cv2.findHomography(self.src_points.astype(np.float32), self.landmark_points.astype(np.float32), cv2.RANSAC,5.0)

        if self.M is not None:
            print("Homography Matrix (Hv):")
            print(self.M)
            #print(self.H)
        else:
            print("Homography matrix is not set.")

        if interpolation == None:
            self.interpolation = cv2.INTER_CUBIC
        else:
            self.interpolation = interpolation

    # Now only Basic grid, needs to be more intelligent
    def draw_grid(self, img):
        #grid_size = 50  # Grid size in pixel
        grid_size = self.config.ppm * 0.5
        
        # Draw vertical grid lines
        for x in range(0, img.shape[1], grid_size):
            cv2.line(img, (x, 0), (x, img.shape[0]), color=(SoccerFieldColors.White.value), thickness=1)
        
        # Draw horizontal grid lines
        for y in range(0, img.shape[0], grid_size):
            cv2.line(img, (0, y), (img.shape[1], y), color=(SoccerFieldColors.White.value), thickness=1)


    def rotate_image(self, img, angle):
        # Compute the center of the image
        center = (img.shape[1] // 2, img.shape[0] // 2)

        # Compute the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

        # Perform the rotation
        rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        return rotated_img
            
    #def warp(self, img):
    def warp(self, src_img, dst_img):

        print(f"Executing warpPerspective, src_width={src_img.shape[1]}, src_height={src_img.shape[0]}, dst_width={dst_img.shape[1]}, dst_height={dst_img.shape[0]}")

        self.plan_view = cv2.warpPerspective(
            src_img, self.M, (dst_img.shape[1], dst_img.shape[0])
        )

        # Print the actual resulting image size after resizing or not
        print(f"Result after warp: width={self.plan_view.shape[1]}, height={self.plan_view.shape[0]}")
        rgb_view_image = cv2.cvtColor(self.plan_view, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{self.config.tmp_data}/_after_warp.jpg', rgb_view_image)

        # Merge both src and dst image
        #self.merged = self.merge_views(src_img,dst_img, self.plan_view)
        self.merged = self.merge_views(dst_img, self.plan_view)

        # Print the actual resulting image size after resizing or not
        print(f"Result after merging: width={self.merged.shape[1]}, height={self.merged.shape[0]}")
        rgb_merged_image = cv2.cvtColor(self.merged, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{self.config.tmp_data}/_after_merge.jpg', rgb_merged_image)

        # Rotate the resulting image by 45 degree
        #result_img = self.rotate_image(result_img, angle=-45)

        # Draw grid on the resulting image
        #self.draw_grid(result_img)

        return self.merged
    
    def merge_views(self, dst, view, blend_factor=0.6):
        # Ensure blend_factor is within the valid range
        if not 0 <= blend_factor <= 1:
            raise ValueError("Blend factor must be between 0 and 1.")
        
        # Ensure both frames have the same shape
        if dst.shape != view.shape:
            raise ValueError("The 'dst' and 'view' frames must have the same dimensions and number of channels.")
        
        # Convert frames to float for accurate computation and to avoid overflow
        dst_float = dst.astype(np.float32)
        view_float = view.astype(np.float32)
        
        # Calculate the weighted blend of the frames
        blended_view = (dst_float * (1 - blend_factor)) + (view_float * blend_factor)
        
        # Convert blended frame back to original data type (e.g., uint8)
        return blended_view.astype(dst.dtype)