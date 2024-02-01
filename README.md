# Camera Calibration and Perspective Correction Tool

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

Tested on Ubuntu 22.04 LTS

1. Clone the repository:

    ```bash
    git clone https://github.com/DXM123/camera_calibration.git
    ```

2. Ensure OpenCV is installed. If not, install it using:

    ```bash
    sudo apt-get install python3-opencv
    ```

3. Install other required packages (`numpy`, `PyQt5` and PyQt5 dependency `libxcb-xinerama0`):

    ```bash
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
    ```

4. Run the script. Make sure it is executable:

    ```bash
    chmod +x Falcons_DeWarp_BETA.py
    ./Falcons_DeWarp_BETA.py
    ```

5. Or install the package in python

    ```bash
    cd camera_calibration
    pip install -e .
    python -c "from falcons.DeWarp import run; run()"
    ```