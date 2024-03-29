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
        # in case of resizing of frame?

        assert isinstance(width, (int, float)) or not isinstance(
            height, (int, float)
        ), "Width and height should be numerical values."
        assert isinstance(supersample, (int, float)), "Supersample should be a numerical value."

        self.supersample = supersample
        
        #Give warper Class the width and height attrbutes
        self.width = width
        self.height = height

        self.points = points
        self.landmark_points = landmark_points

        logging.info(
            "Following values used as input for Warper Class:"
            f" \nWidth: {self.width}.\nHeight: {self.height}.\nSuperSample: {self.supersample}\n"
            f"Points: {self.points}.\nLandMarkPoints:{self.landmark_points}."
        )

        self.M = cv2.getPerspectiveTransform(self.points.astype(np.float32), self.landmark_points.astype(np.float32))
        self.dst = None
        if interpolation == None:
            self.interpolation = cv2.INTER_CUBIC
        else:
            self.interpolation = interpolation

    def warp(self, img, out=None):
        # TODO what's going on here?
        if self.dst is None:
            self.dst = cv2.warpPerspective(img, self.M, (self.width * self.supersample, self.height * self.supersample))
        else:
            self.dst[:] = cv2.warpPerspective(
                img, self.M, (self.width * self.supersample, self.height * self.supersample)
            )

        # TODO is this needed ??? Supersample is a factor by which the image is scaled up for the transformation. This can help reduce aliasing.
        if self.supersample == 1:
            if out == None:
                return self.dst
            else:
                out[:] = self.dst
                return out
        else:
            if out == None:
                return cv2.resize(self.dst, (self.width, self.height), interpolation=self.interpolation)
            else:
                out[:] = cv2.resize(self.dst, (self.width, self.height), interpolation=self.interpolation)
                return out

    # Not working properly yet
    def warp_simple(self, img):
        # Always compute the warped perspective.
        dst = cv2.warpPerspective(img, self.M, (int(self.width * self.supersample), int(self.height * self.supersample)), interpolation=self.interpolation)
        
        # Resize if supersampling is applied.
        if self.supersample != 1:
            self.interpolation = cv2.INTER_CUBIC
            dst = cv2.resize(dst, (int(self.width), int(self.height)), interpolation=self.interpolation)
        
        return dst
