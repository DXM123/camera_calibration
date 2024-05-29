# Falcons Calibration GUI - Work in Progress

## Overview

This Python application provides a user-friendly tool for camera calibration and perspective correction. It uses the OpenCV library for computer vision tasks and PyQt5 for the graphical user interface.

## Features

- **Camera Calibration Tab:**
  - Capture checkerboard pattern images for precise camera calibration.
  - Adjustable settings for checkerboard dimensions (columns, rows, square size).
  - Real-time camera feed displaying detected checkerboard corners.
  - Option to save images with detected corners for subsequent calibration.

- **Calibration Process:**
  - Automated detection of corners in calibration images.
  - Visualization and saving of calibration results, including the camera matrix and distortion coefficients.

- **Testing and Saving Calibration:**
  - Apply calibration to a live camera feed for immediate feedback.
  - Persist calibration parameters as a JSON file for future use.

- **Importing Calibration Data:**
  - Import existing calibration parameters from a JSON file.
  - Apply previously saved calibration data, bypassing the need for recalibration.

- **Perspective Warping Functionality:**
  - Load images for perspective correction.
  - Interactive selection of landmarks for perspective transformation.
  - Live preview of perspective-corrected images.

- **Tuning and Enhancing Images:**
  - Fine-tune perspective warping parameters for optimal results. Use keys 1,2,3 and 4 to select landmark and arrow keys to tune.
  - Saving the results to binary file (TODO)

## Installation and Setup

Tested on Ubuntu 22.04 LTS (Should work on Ubuntu 20.04 LTS also)

1. Clone the repository:

    ```bash
    git clone https://github.com/DXM123/camera_calibration.git
    ```

2. Ensure OpenCV is installed. If not, install it using:

    ```bash
    sudo apt-get install python3-opencv
    ```

3. Install other required packages (`numpy`, `PyQt5`, `pypylon` and PyQt5 dependency `libxcb-xinerama0`):

    ```bash
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
    ```

4. Run the script.

    ```
    ./calibration-gui.py
    ```

**Calibration Example:**

![Screenshot from 2024-05-19 19-57-57](https://github.com/DXM123/camera_calibration/assets/19300348/a798e49d-dc7f-43ab-b824-32e661c66f84)


**Warp Example:**

![Screenshot from 2024-05-29 13-01-29](https://github.com/DXM123/camera_calibration/assets/19300348/36608af4-05b8-4228-ab7a-f63e573e59f3)




