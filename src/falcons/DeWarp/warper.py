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

        # Print the actual resulting image size
        print(f"Resulting image size: width={result_img.shape[1]}, height={result_img.shape[0]}")

        return result_img
