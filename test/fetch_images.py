#!/usr/bin/env python3

import cv2
import time
import os
import subprocess

def fetch_latest_image(remote_path, local_path, remote_host="robocup@r13"):
    """
    Fetches the latest image from a remote host using SCP.

    Args:
    - remote_path: The remote path to the symlink pointing to the latest camera image.
    - local_path: The local path where the image should be saved.
    - remote_host: The username and hostname for the SSH connection.
    """
    scp_command = f"scp {remote_host}:{remote_path} {local_path}"
    subprocess.run(scp_command, shell=True)

def display_camera_stream(local_image_path, window_name="Camera Stream"):
    """
    Displays the camera stream of the locally saved image.

    Args:
    - local_image_path: The path to the local image that is being updated.
    - window_name: The name of the window where the image will be displayed.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        # Read the updated image
        img = cv2.imread(local_image_path)

        # If the image is read successfully, display it
        if img is not None:
            cv2.imshow(window_name, img)
        else:
            print(f"Failed to read image from {local_image_path}")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Wait for a short period to simulate video stream (adjust as needed)
        time.sleep(0.1)  # Adjust the sleep time to control frame rate

    cv2.destroyAllWindows()

# Paths and parameters
remote_image_path = "/dev/shm/cam0.jpg"  # Adjust for the specific camera symlink
local_image_path = "./cam0.jpg"  # Local path where the image will be saved
window_name = "Remote Camera Stream"

# Fetch and display loop
while True:
    fetch_latest_image(remote_image_path, local_image_path)
    display_camera_stream(local_image_path, window_name)
