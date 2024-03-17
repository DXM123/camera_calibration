#!/usr/bin/env python3

import cv2
import numpy as np
import glob
import os

# Run this script in the folder containing the .JPG images
#CHECKERBOARD = (6, 9)
CHECKERBOARD = (7, 7) # columns / # rows

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

_img_shape = None
objpoints = []  # 3d points in real world spaceobjp
imgpoints = []  # 2d points in image plane.
images = glob.glob('*.jpg')

# Loop through images
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue
    if _img_shape is None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."

    # Find Corners using gray scale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS + cv2.CALIB_CB_NORMALIZE_IMAGE) # Current Falcons CamCal option
    
    if ret:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)

        # Visual confirmation of corner detection
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('Found Chessboard Corners', img)
        cv2.waitKey(500)  # Wait for 500 ms

if not objpoints or not imgpoints or len(objpoints) != len(imgpoints):
    print("Error: Mismatch or empty data in object points and image points.")
    print(f"Object points: {len(objpoints)}, Image points: {len(imgpoints)}")
    cv2.destroyAllWindows()
    exit(1)

## Calibrate

N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

try:
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
    print(f"Found {N_OK} valid images for calibration")
    print(f"DIM={_img_shape[::-1]}")
    print(f"K=np.array({K.tolist()})")
    print(f"D=np.array({D.tolist()})")
except cv2.error as e:
    print(f"OpenCV Error during calibration: {e}")

cv2.destroyAllWindows()
