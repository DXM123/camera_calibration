# Camera Calibration and Perspective Correction Tool

## Overview

This Python application provides a user-friendly tool for camera calibration and perspective correction. It uses the OpenCV library for computer vision tasks and PyQt5 for the graphical user interface.

## Features

- **Camera Calibration Tab:**
  - Capture images of a checkerboard pattern for camera calibration.
  - Set the number of columns, rows, and square size of the checkerboard.
  - Live camera feed with detected corners during calibration.
  - Save captured images with detected corners for later calibration.

- **Calibration Process:**
  - Detect corners in calibration images for camera calibration.
  - Display and save calibration results, including camera matrix and distortion coefficients.

- **Testing Calibration:**
  - Test the calibration by applying it to the live camera feed.
  - Save calibration parameters to a file.
 
- **Save Calibration:**
  - Save calibration parameters to a JSON file.
  - Can be imported again.

- **Import Calibration:**
  - Import calibration parameters from a JSON file.
  - Use imported parameters for testing without recalibration.
 
- **Perspective-Warp Tab (TODO):**
  - Load an image for perspective correction (warping).
  - Display the loaded image.
  - Initiate perspective-warp process.

## How to Use - Tested on Ubuntu 22.04 LTS

1. Clone the repository:

    ```bash
    git clone https://github.com/DXM123/camera_calibration.git
    ```
2. Make sure opencv is availble. You can install it with

    ```bash
    sudo apt-get install python3-opencv
    ```
    and install other requirements (`numpy` and `PyQt5`) with

    ```bash
    pip install -r requirements.txt
    ```
4. Run the script (make executable first)
