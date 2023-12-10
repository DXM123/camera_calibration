#!/usr/bin/env python3

import cv2
import numpy as np

points = []
selected_point = 0  # Index of the selected point

def adjust_point(point, key):
    """ Adjust the point based on arrow key input """
    if key == 82:  # Up arrow
        return (point[0], point[1] - 1)
    elif key == 84:  # Down arrow
        return (point[0], point[1] + 1)
    elif key == 81:  # Left arrow
        return (point[0] - 1, point[1])
    elif key == 83:  # Right arrow
        return (point[0] + 1, point[1])
    return point

# Mouse callback function
def mouse_click(event, x, y, flags, param):
    global points, bgimage
    if event == cv2.EVENT_LBUTTONUP:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(bgimage, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, bgimage)

            if len(points) == 4:
                print("Four corners selected. Press any key to continue.")

# Load the image
img = cv2.imread('Test-Pwarp.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"

# Create a copy of the image for displaying selected points
bgimage = img.copy()

# Set up the window for the original frame
window_name = "Original Frame with Landmarks"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_click)
cv2.imshow(window_name, bgimage)

# Create a separate window for the adjusted frame
adjusted_window_name = "Adjusted Frame"
cv2.namedWindow(adjusted_window_name)

# Wait until four corners are selected or 'ESC' is pressed
while len(points) < 4:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'ESC' to exit if needed
        break

# Allow adjustment of points with arrow keys
while True:
    key = cv2.waitKey(0)
    if key == 27:  # ESC key to exit
        break
    elif key == 9:  # TAB key to select next point
        selected_point = (selected_point + 1) % 4
    else:
        points[selected_point] = adjust_point(points[selected_point], key)

        # Update the transformation and display
        pts1 = np.float32(points)
        h, w, _ = img.shape
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (w, h))

        # Display updated image with selected points
        cv2.imshow(adjusted_window_name, dst)

        # Update the selected point circle to red in the original frame
        bgimage = img.copy()
        for i, point in enumerate(points):
            if i == selected_point:
                cv2.circle(bgimage, (point[0], point[1]), 5, (0, 0, 255), -1)
            else:
                cv2.circle(bgimage, (point[0], point[1]), 5, (0, 255, 0), -1)

        cv2.imshow(window_name, bgimage)

# Close all OpenCV windows
cv2.destroyAllWindows()
