from typing import Union
import logging

import cv2
import numpy as np

class Warper(object):
    def __init__(
        self,
        points: np.ndarray,
        landmark_points: np.ndarray,
        width: Union[int, float] = 640,
        height: Union[int, float] = 480,
        supersample: Union[int, float] = 2,
        interpolation=None,
    ):
        # TODO The types of width and height were inconsistent with their default
        # I fixed them now to be `Union[int, float]`. Do they have to be dynamic
        # in case of resizing of frame? -> Yes, just initial staring values

        assert isinstance(width, (int, float)) or not isinstance(
            height, (int, float)
        ), "Width and height should be numerical values."
        assert isinstance(supersample, (int, float)), "Supersample should be a numerical value."
        
        #Give warper Class the attrbutes
        self.width = width
        self.height = height
        self.supersample = supersample
        self.points = points
        self.landmark_points = landmark_points
        self.dst = None

        logging.info(
            "Following values used as input for Warper Class:"
            f" \nWidth: {self.width}.\nHeight: {self.height}.\nSuperSample: {self.supersample}\n"
            f" \nWidth: {self.width}.\nHeight: {self.height}.\nSuperSample: {supersample}\n"
            f"Points: {self.points}.\nLandMarkPoints:{self.landmark_points}."
        )

        self.M = cv2.getPerspectiveTransform(self.points.astype(np.float32), self.landmark_points.astype(np.float32))

        if interpolation == None:
            self.interpolation = cv2.INTER_CUBIC
        else:
            self.interpolation = interpolation

    # Now only Basic grid, needs to be more intelligent
    # frame, Hv=M, boundaries_rcs_mm = ??
    def draw_grid(self, img):
        grid_size = 50  # Grid size in pixels
        
        # Draw vertical Green lines
        for x in range(0, img.shape[1], grid_size):
            cv2.line(img, (x, 0), (x, img.shape[0]), color=(255, 255, 255), thickness=1)
        
        # Draw horizontal Green lines
        for y in range(0, img.shape[0], grid_size):
            cv2.line(img, (0, y), (img.shape[1], y), color=(255, 255, 255), thickness=1)

    ##########################################################################

    def draw_grid_new(self, img, boundaries_rcs_mm=None):
        MM = 1e-3  # Define MM as per the previous context
        MX, MY = 10, 10
        Hv = self.M  # Using the homography matrix of the Warper instance
        
        # Green 1m-per-square grid
        for rx in range(-MX, MX + 1):
            iPts = np.array([[rx / MM, 0], [rx / MM, MY / MM]], dtype=np.float32)
            oPts = cv2.perspectiveTransform(np.array([iPts]), Hv)[0]
            cv2.line(img, tuple(oPts[0]), tuple(oPts[1]), (0, 255, 0), 1)
        
        for ry in range(MY + 1):
            iPts = np.array([[-MX / MM, ry / MM], [MX / MM, ry / MM]], dtype=np.float32)
            oPts = cv2.perspectiveTransform(np.array([iPts]), Hv)[0]
            cv2.line(img, tuple(oPts[0]), tuple(oPts[1]), (0, 255, 0), 1)
        
        # Red diagonals
        for sign in [-1, 1]:
            iPts = np.array([[sign * MX / MM, MY / MM], [0, 0]], dtype=np.float32)
            oPts = cv2.perspectiveTransform(np.array([iPts]), Hv)[0]
            cv2.line(img, tuple(oPts[0]), tuple(oPts[1]), (255, 0, 0), 1)
        
        # Red field boundaries (if provided)
        if boundaries_rcs_mm is not None:
            for i in range(1, len(boundaries_rcs_mm)):
                iPts = np.array([
                    [boundaries_rcs_mm[i].x, boundaries_rcs_mm[i].y],
                    [boundaries_rcs_mm[i-1].x, boundaries_rcs_mm[i-1].y]
                ], dtype=np.float32)
                oPts = cv2.perspectiveTransform(np.array([iPts]), Hv)[0]
                cv2.line(img, tuple(oPts[0]), tuple(oPts[1]), (255, 0, 0), 1)
    
    ##########################################################################

    def rotate_image(self, img, angle):
        # Compute the center of the image
        center = (img.shape[1] // 2, img.shape[0] // 2)

        # Compute the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

        # Perform the rotation
        rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        return rotated_img

            
    def warp(self, img):
        # Determine if resizing is necessary based on supersample value
        # Seems we need resizing anyway, when set to 1 scaling is wrong
        # Now set in widget to self.supersample = 2
        resizing_needed = self.supersample != 1

        print(f"Executing warpPerspective with supersample={self.supersample}, width={self.width}, height={self.height}, resulting size=({int(self.width * self.supersample)}, {int(self.height * self.supersample)})")
        self.dst = cv2.warpPerspective(
            img, self.M, (self.width * self.supersample, self.height * self.supersample)
        )
        
        # Describe the action being taken based on whether resizing is needed
        action_description = "Resizing required" if resizing_needed else "No resizing needed"
        print(f"{action_description} Super Sample is: {self.supersample}")

        # Perform the necessary action: resizing if needed, or directly returning the warped image
        if resizing_needed:
            # Resize the image if supersampling is not equal to 1
            print(f"Resizing image to width={self.width}, height={self.height} with interpolation={self.interpolation}")
            result_img = cv2.resize(self.dst, (int(self.width), int(self.height)), interpolation=self.interpolation)
        else:
            # No resizing needed, use the warped image directly -> But odd output
            result_img = self.dst

        # Rotate the resulting image by 45 degrees
        result_img = self.rotate_image(result_img, angle=-45)

        # Print the actual resulting image size
        print(f"Resulting image size: width={result_img.shape[1]}, height={result_img.shape[0]}")

        #Plot Grid here ? TODO
        # Draw grid on the resulting image
        self.draw_grid(result_img)

        return result_img
