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
        matrix: np.ndarray = None,
        dist_coeff: np.ndarray = None,
        interpolation=None,
    ):
        self.config = get_config()
        self.D = dist_coeff
        self.K = matrix

        # TEST
        print(f"Camera Distortion Coefficients: {self.D}")
        print(f"Camera Matrix: {self.K}")

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
        #self.dst = None
        self.Hv = None

        #logging.info(
        print(
            "Following values used as input for Warper Class:"
            f" \nWidth: {self.src_width}.\nHeight: {self.src_height}.\n"
            f"Points: {self.src_points}.\nLandMarkPoints:{self.landmark_points}."
        )

        # Two way to calculate the Homography Hv (output is identical but one also produces mask (not used now))
        self.Hv = cv2.getPerspectiveTransform(self.src_points.astype(np.float32), self.landmark_points.astype(np.float32))
        #self.Hv, mask = cv2.findHomography(self.src_points.astype(np.float32), self.landmark_points.astype(np.float32), cv2.RANSAC,5.0)

        if self.Hv is not None:
            print("Homography Matrix (Hv):")
            print(self.Hv)
        else:
            print("Homography Matrix (Hv) is not set.")

        if interpolation == None:
            self.interpolation = cv2.INTER_CUBIC
        else:
            self.interpolation = interpolation
            

    def warp(self, src_img, dst_img):

        print(f"Executing warpPerspective, src_width={src_img.shape[1]}, src_height={src_img.shape[0]}, dst_width={dst_img.shape[1]}, dst_height={dst_img.shape[0]}")

        self.plan_view = cv2.warpPerspective(
            src_img, self.Hv, (dst_img.shape[1], dst_img.shape[0])
        )

        # Print the actual resulting image size after resizing or not
        print(f"Result after warp: width={self.plan_view.shape[1]}, height={self.plan_view.shape[0]}")
        rgb_view_image = cv2.cvtColor(self.plan_view, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{self.config.tmp_data}/_after_warp.jpg', rgb_view_image)

        # Merge both src and dst image
        self.merged = self.merge_views(dst_img, self.plan_view)

        # Print the actual resulting image size after resizing or not
        print(f"Result after merging: width={self.merged.shape[1]}, height={self.merged.shape[0]}")
        rgb_merged_image = cv2.cvtColor(self.merged, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{self.config.tmp_data}/_after_merge.jpg', rgb_merged_image)

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
    
    ################################ TEST LUT ##############################

    def transform_point(self, x, y):
        Hv = self.Hv
        # Create the homogeneous coordinate of the point
        point = np.array([x, y, 1])
        
        # Apply the transformation
        transformed_point = np.dot(Hv, point)
        
        # Normalize if the last component is not 1
        if transformed_point[2] != 0:
            transformed_point /= transformed_point[2]
        
        #return transformed_point
        return transformed_point[:2]

    def undistort_point(self, x, y):
        # Print the distortion coefficients and camera matrix
        #print("Distortion Coefficients:", self.D)
        #print("Camera Matrix:", self.K)


        # Correct the point for lens distortion
        distorted = np.array([[[x, y]]], dtype=np.float32)
        #print("Distorted Point:", distorted)

        undistorted = cv2.fisheye.undistortPoints(distorted, self.K, self.D, P=self.K)
        #print("Undistorted Point:", undistorted[0,0])

        return undistorted[0,0]

    def create_lookup_table(self, img_shape):
        height, width = img_shape
        lut = np.zeros((height, width, 2), dtype=np.float32)

        # Add Camera Calibration
        #lut = self.camera_correct(self, lut)

        for y in range(height):
            for x in range(width):
                # Apply lens correction
                x_corr, y_corr = self.undistort_point(x, y)
                # Transform the corrected coordinates
                x_prime, y_prime = self.transform_point(x_corr, y_corr)
                lut[y, x] = [x_prime, y_prime]

        return lut