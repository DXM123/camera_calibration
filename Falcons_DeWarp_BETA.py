#!/usr/bin/env python3

import sys
import cv2
import datetime
import os
import json # OOB available in python (to save calibration file)
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QPushButton,
    QWidget,
    QAction,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFileDialog,
    QStatusBar,
    QFrame,
    QLineEdit,
    QMessageBox,
    QSizePolicy,
    QTextEdit,
    QRadioButton,
    QButtonGroup,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSlot, pyqtSignal, Qt, QTimer

# Dependency OpenCV 4.x, PyQT5 + Numpy (Default in UBNT 22.04)
# TODO Default disable tab2, only enable after finish Calibartion or loading Calibration JSON
# TODO Add logic for usb, and network
# TODO Optimize landmark selection selections
# TODO Save to binairy cv mat file

#Config Default Variables - Enter their values according to your Checkerboard, normal 64 (8x8) -1 inner corners only
no_of_columns = 7  #number of columns of your Checkerboard
no_of_rows = 7  #number of rows of your Checkerboard
square_size = 27.0 # size of square on the Checkerboard in mm -> This is no longer required?
min_cap = 3 # minimum or images to be collected by capturing (Default is 10), minimum is 3

# Assuming the soccer field is 22 x 14 meters - old
soccer_field_width = 22
soccer_field_length = 14

# Field Size and other dimensions for MSL field 18 x 12 meter 
field_length = 18  # meters
field_width = 12  # meters
penalty_area_length = 2.25  # E, meters
penalty_area_width = 6.5  # C, meters
goal_area_length = 0.75  # F, meters
goal_area_width = 3.5  # D, meters
center_circle_radius = 2  # H, meters
spot_radius = 0.15
goal_depth = 0.5  # Goal depth,
goal_width = 2.0  # Goal width 2m for this field -> 2.4m allowed?
line_width = 0.125  # K, meters
ppm = 100  # pixels per meter
safe_zone = 1  # Safety zone around the field, meters

### Total Field Size
field_length_total = field_length + 2 * safe_zone  # Adding safety zone to the length
field_width_total = field_width + 2 * safe_zone  # Adding safety zone to the width

# Define basic colors for Soccer Field (RGB)
black = (0, 0, 0)
white = (255, 255, 255)
gray = (128, 128, 128)
red = (255, 0, 0)
green = (0, 255, 0)
lightgreen = (144, 238, 144)
darkgreen = (0, 100, 0)
blue = (0, 0, 255)
lightblue = (173, 216, 230)
pink = (255, 192, 203)
magenta = (255, 0, 255)

# For 18 x 12 meter Field
# Landmark 1, where the middle circle meets the middle line
landmark1 = (2,0)

# Landmark 2, where the middle line meets the outer field line
landmark2 = (6,0) # field_width / 2, 0

# Landmark 3, from the center circle spot towards the goal, where the (fictive) line meets the center circle line
landmark3 = (0,2) # 0, center_circle_radius

# # Landmark 4, Penalty Spot
landmark4 = (0,6) # 0, field_lenght / 2 - penalty_spot (3)

# Center Spot 0,0 FCS
FCS0 = (int(field_length_total / 2) * ppm, int(field_width_total / 2) * ppm )
#print(f"FCS 0,0 {FCS0}")

# Marker Spot 0,2 FCS
FCS02 = (int((field_length_total / 2) * ppm) - int(2 * ppm), int(field_width_total / 2) * ppm )
#print(f"FCS 0,2 {FCS02}")

# Marker Spot 0,6 FCS - Penalty spot left
FCS06 = (int((field_length_total / 2) * ppm) - int(6 * ppm), int(field_width_total / 2) * ppm )
#print(f"FCS 0,6 {FCS06}")

# Marker Spot 2,0 FCS
FCS20 = (int(field_length_total / 2) * ppm, int((field_width_total / 2) * ppm) - int(2 * ppm))
#print(f"FCS 2,0 {FCS20}")

# Marker Spot 6,0 FCS
FCS60 = (int(field_length_total / 2) * ppm, int((field_width_total / 2) * ppm) - int(6 * ppm))
#print(f"FCS 6,0 {FCS60}")

# Marker Spot -2,0 FCS
FCSmin20 = (int(field_length_total / 2) * ppm, int((field_width_total / 2) * ppm) + (2 * ppm))
#print(f"FCS -2,0 {FCSmin20}")

# Marker Spot -6,0 FCS
FCSmin60 = (int(field_length_total / 2) * ppm, int((field_width_total / 2) * ppm) + (6 * ppm))
#print(f"FCS -6,0 {FCSmin60}")

# Marker Spot 0,-2 FCS
FCS0min2 = (int(field_length_total / 2) * ppm, int((field_width_total / 2) * ppm) - (2 * ppm))
#print(f"FCS 0,-2 {FCS0min2}")

# Marker Spot 0,-6 FCS
FCS0min6 = (int(field_length_total / 2) * ppm, int((field_width_total / 2) * ppm) - (6 * ppm))
#print(f"FCS 0,-6 {FCS0min6}")

################## Field Drawing Class ####################

class Draw_SoccerField:
    def __init__(self):
        self.length = field_length + 2 * safe_zone  # Adding safety zone to the length
        self.width = field_width + 2 * safe_zone  # Adding safety zone to the width
        self.line_width = line_width
        self.center_circle_radius = center_circle_radius
        self.ppm = ppm
        self.safe_zone = safe_zone
        self.spot_radius = spot_radius

        # Create a blank image (dark green background)
        self.field_image = np.full((int(self.width * self.ppm), int(self.length * self.ppm), 3), darkgreen, dtype=np.uint8)

    def draw_line(self, start_point, end_point):
        thickness = int(self.line_width * self.ppm)
        start_pixel = (int(start_point[0] * self.ppm), int(start_point[1] * self.ppm))
        end_pixel = (int(end_point[0] * self.ppm), int(end_point[1] * self.ppm))
        cv2.line(self.field_image, start_pixel, end_pixel, (white), thickness)

    def draw_circle(self, center, radius):
        center_pixel = (int(center[0] * self.ppm), int(center[1] * self.ppm))
        cv2.circle(self.field_image, center_pixel, int(radius * self.ppm), (white), int(self.line_width * self.ppm))

    def draw_spot(self, center, radius):
        center_pixel = (int(center[0] * self.ppm), int(center[1] * self.ppm))
        cv2.circle(self.field_image, center_pixel, int(radius * self.ppm), (white), int(self.line_width * self.ppm))

    def draw_rectangle(self, top_left, bottom_right):
        top_left_pixel = (int(top_left[0] * self.ppm), int(top_left[1] * self.ppm))
        bottom_right_pixel = (int(bottom_right[0] * self.ppm), int(bottom_right[1] * self.ppm))
        cv2.rectangle(self.field_image, top_left_pixel, bottom_right_pixel, (white), int(self.line_width * self.ppm))

    def draw_goal(self, center, width, depth):
        # Goals are drawn as rectangles perpendicular to the field's length
        half_width = width / 2
        top_left = (center[0] - depth, center[1] - half_width)
        bottom_right = (center[0], center[1] + half_width)
        self.draw_rectangle(top_left, bottom_right)

    def generate_field(self):
        # Draw the safety zone
        self.draw_rectangle((self.safe_zone, self.safe_zone), (self.length - self.safe_zone, self.width - self.safe_zone))

        # Offset all field elements by the safety zone
        offset = self.safe_zone

        # Drawing the outline of the field
        self.draw_rectangle((offset, offset), (self.length - offset, self.width - offset))

        # Drawing the center line
        self.draw_line((self.length / 2, offset), (self.length / 2, self.width - offset))

        # Drawing the center circle
        self.draw_circle((self.length / 2, self.width / 2), self.center_circle_radius)

        # Drawing the penalty areas
        # Only the x-coordinate (left and right positions) is adjusted by the offset
        self.draw_rectangle((offset, (self.width - penalty_area_width) / 2), (penalty_area_length + offset, (self.width + penalty_area_width) / 2))
        self.draw_rectangle((self.length - penalty_area_length - offset, (self.width - penalty_area_width) / 2), (self.length - offset, (self.width + penalty_area_width) / 2))

        # Drawing the goal areas
        # Only the x-coordinate (left and right positions) is adjusted by the offset
        self.draw_rectangle((offset, (self.width - goal_area_width) / 2), (goal_area_length + offset, (self.width + goal_area_width) / 2))
        self.draw_rectangle((self.length - goal_area_length - offset, (self.width - goal_area_width) / 2), (self.length - offset, (self.width + goal_area_width) / 2))

        # Drawing the goals
        # The goals are drawn at the start and end of the field, adjusted by the offset
        self.draw_goal((0 + offset, self.width / 2), goal_width, goal_depth)
        self.draw_goal((self.length - goal_depth, self.width / 2), goal_width, goal_depth)

        # Drawing Center Spot 0,0 FCS
        self.draw_spot((self.length / 2, self.width / 2), self.spot_radius )

        return self.field_image

################# Perspective Warp Classes ################# 

class Warper(object):
    def __init__(self,points,width=640,height=480,supersample=2,interpolation=None):
        self.points = points
        self.width  = width
        self.height = height
        self.supersample = supersample

        print("Following values used as input for Warper Class: ")
        print(f"W:{self.width()}")  # Access the width attribute of the warper instance
        print(f"H:{self.height()}")  # Access the height attribute of the warper instance
        print(f"SS:{self.supersample}")  # Access the supersample attribute of the warper instance

        # Collected points
        self.pts1 = np.float32([points[0],points[1],points[3],points[2]])

        print("self.pts1:", self.pts1)

        # Call the height and width method to get the actual value of the frame
        W = self.width()
        H = self.height()

        # Check if W and H are integers or floats
        if not isinstance(W, (int, float)) or not isinstance(H, (int, float)):
            raise ValueError("Width and height should be numerical values.")

        # Check if supersample is a numerical value
        if not isinstance(supersample, (int, float)):
            raise ValueError("Supersample should be a numerical value.")

        # Update self.pts2 with the FCS landmarks 2,0 6,0 0,2 0,6 
        # self.pts2 = np.float32([FCS02, FCS06, FCS60, FCS20]) # mirror
        self.pts2 = np.float32([FCS20, FCS60, FCS06, FCS02]) # Odd order ofcourse but looks ok TODO

        print("self.pts2:", self.pts2)

        self.M = cv2.getPerspectiveTransform(self.pts1,self.pts2)
        self.dst = None
        if (interpolation == None):
            self.interpolation = cv2.INTER_CUBIC
        else:
            self.interpolation = interpolation

    def warp(self,img,out=None):
        # Call the height and width method to get the actual value of the frame
        W = self.width()
        H = self.height()

        M = self.M
        supersample = self.supersample

        if self.dst is None:
            self.dst = cv2.warpPerspective(img,M,(W*supersample,H*supersample))
        else:
            self.dst[:] = cv2.warpPerspective(img,M,(W*supersample,H*supersample))

        # is this needed ??? TODO
        if supersample == 1:
            if out == None:
                return self.dst
            else:
                out[:] = self.dst
                return out
        else:
            if out == None:
                return cv2.resize(self.dst, (W,H), interpolation=self.interpolation)
            else:
                out[:] = cv2.resize(self.dst, (W,H), interpolation=self.interpolation)
                return out

###############################################################

class CameraWidget (QWidget):
    # Add a signal for updating the status
    update_status_signal = pyqtSignal(str)

    # Set the initial countdown time to save images
    countdown_seconds = 3 # 3 seconds feels long

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)

         # Add a QTimer countdown timer
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        
        # Add a QTimer for continuous camera feed updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_feed)

        # Add a flag to track state
        self.capture_started = False # Track if caputure is started
        self.test_started = False # Track if test is started
        self.cal_imported = False # Track if imported calibration is used
        self.image_dewarp = False # Track if an image is used for dewarping
        self.usb_dewarp = False # Track if an USB camera is used for dewarping
        self.network_dewarp = False # Track if an network stream is used for dewarping
        self.image_tuning_dewarp = False # Track if an image tuning is used for dewarping

        # Need the camera object in this Widget
        self.cap = cv2.VideoCapture(0) # webcam object
        self.pixmap = None

        # Add cv_image Temp
        self.cv_image = None

        # Add field_image Temp
        self.field_image = None

        # Placeholder to count the ammount of images saved
        self.ImagesCounter = 0

        # Initialize camera_matrix as None
        self.camera_matrix = None

        # Initialize camera_distortion coefficients as None
        self.camera_dist_coeff = None

        # Do i need rvecs and tvecs?

        # Define basic line thickness to draw soccer field
        self.line_thickness = 2

        # Set variable to store selected objectpoint for dewarp
        self.points = []
        self.selected_point = 0  # Index of the selected point for tuning dewarp

         # Add the supersample attribute
        self.supersample = 2 

        #..:: Start UI layout ::..#

        # Use a central layout for the entire widget
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        #self.tabs.setFocusPolicy(Qt.ClickFocus)  # or Qt.StrongFocus
        self.tabs.setFocusPolicy(Qt.FocusPolicy.StrongFocus) # TEST TODO

        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab2.setFocusPolicy(Qt.FocusPolicy.StrongFocus) # TEST TODO
        self.tabs.resize(300, 200)

        # Add tabs
        self.tabs.addTab(self.tab1, "Camera Calibration")
        self.tabs.addTab(self.tab2, "Perspective-Warp")

        # ..::Create first tab::.. #

        # Create Horizontal Box Layout for tab 1
        self.tab1.layout = QHBoxLayout(self.tab1)

        # Create Vertical Box Layout for tab1 inner frame
        self.tab1inner = QWidget()
        self.tab1inner.layout = QVBoxLayout(self.tab1inner)

        # Add Start Capture on top
        self.captureButton1 = QPushButton("Start Capture", self.tab1inner)
        self.captureButton1.clicked.connect(self.start_capture)
        self.tab1inner.layout.addWidget(self.captureButton1)

        # Add Camera Frame
        self.cameraFrame = QLabel("Image will be displayed here", self.tab1inner)

        # Set word wrap to ensure text fits within the QLabel width
        self.cameraFrame.setWordWrap(True)

        # Set alignment to center the text within the QLabel
        self.cameraFrame.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.cameraFrame.resize(640, 480)
        self.cameraFrame.setFrameShape(QFrame.Box)
        self.tab1inner.layout.addWidget(self.cameraFrame)

        # Add Done Button last
        self.doneButton1 = QPushButton("DONE", self.tab1inner)
        self.doneButton1.clicked.connect(self.check_done)
        self.tab1inner.layout.addWidget(self.doneButton1)

        # Add tab1inner to tab1
        self.tab1.layout.addWidget(self.tab1inner)

        # Create Vertical Layout for otions frame on right side
        self.optionsFrame = QWidget()
        self.optionsFrame.layout = QVBoxLayout(self.optionsFrame)
        self.optionsFrame.layout.setAlignment(Qt.AlignTop)  # Align the layout to the top
        #self.optionsFrame.layout.addStretch(1)

        # Set fixed width for optionsFrame
        self.optionsFrame.setFixedWidth(400)

        # Add options widgets to optionsFrame:
        option_label = QLabel("Chessboard Options:")
        self.optionsFrame.layout.addWidget(option_label)

        # Add columns option
        self.columnsLabel = QLabel('# of Columns:', self)
        #self.columnsLabel.resize(220, 40)
        self.columnsInput = QLineEdit(str(no_of_columns), self)
        self.optionsFrame.layout.addWidget(self.columnsLabel)
        self.optionsFrame.layout.addWidget(self.columnsInput)

        # Add row option
        self.rowsLabel = QLabel('# of Rows:', self)
        self.rowsInput = QLineEdit(str(no_of_rows), self)
        self.optionsFrame.layout.addWidget(self.rowsLabel)
        self.optionsFrame.layout.addWidget(self.rowsInput)

        # Add square Size option
        self.squareSizeLabel = QLabel('Square size (mm):', self)
        self.squareSizeRow = QLineEdit(str(square_size), self)
        self.optionsFrame.layout.addWidget(self.squareSizeLabel)
        self.optionsFrame.layout.addWidget(self.squareSizeRow)

        # Add output Display (display at start is to small)
        self.outputWindowLabel = QLabel('Output Display:', self)
        self.outputWindow = QTextEdit()
        self.outputWindow.setReadOnly(True)

        # Set the size policy to make the output window stretch vertically
        outputSizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.outputWindow.setSizePolicy(outputSizePolicy)

        # Set the alignment of the output window
        outputWindowAlignment = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        self.outputWindow.setAlignment(outputWindowAlignment)

        self.optionsFrame.layout.addWidget(self.outputWindowLabel)
        self.optionsFrame.layout.addWidget(self.outputWindow)

        # Add optionsFrame to the right side
        self.tab1.layout.addWidget(self.optionsFrame)

        # Set tab1.layout as the layout for tab1
        self.tab1.setLayout(self.tab1.layout)

        # ..::Create second tab::.. #

        # Create Horizontal Box Layout for tab 2
        self.tab2.layout = QHBoxLayout(self.tab2)

        # Create Vertical Box Layout for tab1 inner frame
        self.tab2inner = QWidget()
        self.tab2inner.layout = QVBoxLayout(self.tab2inner)

        # Add Load Image on top (For now load from R8 folder)
        self.loadImage = QPushButton("Load Image", self.tab2inner)
        self.loadImage.clicked.connect(self.load_image)

        self.tab2inner.layout.addWidget(self.loadImage)

        # Add Image Frame 
        self.imageFrame = QLabel("Image will be displayed here", self.tab2inner)

        # Set word wrap to ensure text fits within the QLabel width
        self.imageFrame.setWordWrap(True)

        # Set alignment to center the text within the QLabel
        self.imageFrame.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.imageFrame.setFrameShape(QFrame.Box)

        # Set focus on ImageFrame to receive key events --> TEST TAB -> No Difference!! TODO
        self.imageFrame.setFocusPolicy(Qt.StrongFocus)
        self.tab2inner.layout.addWidget(self.imageFrame)

        # Add Start De-warp Button last
        self.startButtonPwarp = QPushButton("START Perspective-Warp", self.tab2inner)
        self.startButtonPwarp.clicked.connect(self.start_pwarp)
        self.tab2inner.layout.addWidget(self.startButtonPwarp)

        # Add tab1inner to tab2
        self.tab2.layout.addWidget(self.tab2inner)

        # Create Vertical Layout for processs frame on right side
        self.ProcessFrame = QWidget()
        self.ProcessFrame.layout = QVBoxLayout(self.ProcessFrame)
        self.ProcessFrame.layout.setAlignment(Qt.AlignTop)  # Align the layout to the top
        #self.ProcessFrame.layout.addStretch(1)

        # Set fixed width for optionsFrame
        self.ProcessFrame.setFixedWidth(500)

        # Add options widgets to optionsFrame:
        process_label = QLabel("Process Options:")
        process_label.setFixedHeight(48)
        self.ProcessFrame.layout.addWidget(process_label)

        # Add radio buttons for selecting options
        self.radio_layout = QHBoxLayout()

        self.radio_usb = QRadioButton("USB")
        self.radio_network = QRadioButton("Network")
        self.radio_image = QRadioButton("Image")

        # Set "USB" as the default option
        self.radio_image.setChecked(True)

        # Connect the toggled signal of radio buttons to a slot function
        self.radio_usb.toggled.connect(self.update_load_button_state)
        self.radio_network.toggled.connect(self.update_load_button_state)

        # Create a button group to make sure only one option is selected at a time
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.radio_usb)
        self.button_group.addButton(self.radio_network)
        self.button_group.addButton(self.radio_image)

        self.radio_layout.addWidget(self.radio_usb)
        self.radio_layout.addWidget(self.radio_network)
        self.radio_layout.addWidget(self.radio_image)

        self.ProcessFrame.layout.addLayout(self.radio_layout)

        # Add Process Frame 
        self.ProcessImage = QLabel("Processed output will be displayed here", self.ProcessFrame)

        # Set word wrap to ensure text fits within the QLabel width
        self.ProcessImage.setWordWrap(True)

        # Set alignment to center the text within the QLabel
        self.ProcessImage.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.ProcessImage.setFrameShape(QFrame.Box)

        # Set the aspect ratio of soccer field
        aspect_ratio = soccer_field_width / soccer_field_length

        # Set the height of the QLabel to fixed pixels
        label_height = 300

        # Draw and display soccer field in the ProcessFrame
        field_drawer = Draw_SoccerField()
        soccer_field_image = field_drawer.generate_field()

        # TEMP store it
        self.field_image = soccer_field_image

        # Convert to Pixmap
        soccer_field_pixmap = self.imageToPixmap(soccer_field_image) 

        if soccer_field_pixmap:
             # Load the image using QPixmap
            pixmap = QPixmap(soccer_field_pixmap)
            self.ProcessImage.setPixmap(pixmap)
            self.ProcessImage.setScaledContents(True) # needed or field wont fit

        # Calculate the corresponding width based on the aspect ratio
        label_width = int(label_height * aspect_ratio)

        # Set size policy and fixed size
        self.ProcessImage.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ProcessImage.setFixedSize(label_width, label_height)

        self.ProcessFrame.layout.addWidget(self.ProcessImage)
        
        # Add process Output Display (display at start is to small)
        self.processOutputWindowLabel = QLabel('Output Display:', self)
        self.processoutputWindow = QTextEdit()
        self.processoutputWindow.setReadOnly(True)

        # Set the size policy to make the process output window stretch vertically
        processoutputSizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.processoutputWindow.setSizePolicy(processoutputSizePolicy)

        # Set the alignment of the process output window
        processoutputWindowAlignment = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        self.processoutputWindow.setAlignment(processoutputWindowAlignment)

        # Also add output window here
        self.ProcessFrame.layout.addWidget(self.processOutputWindowLabel)
        self.ProcessFrame.layout.addWidget(self.processoutputWindow)

        # Add optionsFrame to the right side
        self.tab2.layout.addWidget(self.ProcessFrame)

        # Set tab2.layout as the layout for tab2
        self.tab2.setLayout(self.tab2.layout)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def start_capture(self):
        global no_of_columns, no_of_rows, square_size

        if not self.capture_started:
            # Read user-input values for columns, rows, and square size
            no_of_columns = int(self.columnsInput.text())
            no_of_rows = int(self.rowsInput.text())
            square_size = float(self.squareSizeRow.text())

            # Start the timer when the button is clicked
            self.timer.start(100)  # Set the interval in milliseconds (e.g. 100 milliseconds)
            
            # Update button text
            self.captureButton1.setText("Pauze Capture")
            
            # Emit the signal with the updated status text
            self.update_status_signal.emit("Capture in progess...")

            # Start the countdown timer when capture is started
            self.countdown_timer.start(1000)  # Update every 1000 milliseconds (1 second)

        else:
            # Stop the timer when the button is clicked again
            self.timer.stop()

            # Update button text
            self.captureButton1.setText("Resume Capture")

            # Emit the signal with the updated status text
            self.update_status_signal.emit("Capture pauzed")

        # Toggle capture state
        self.capture_started = not self.capture_started
    
    # Check def convert_cvimage)to_pixmap / def imageToPixmap (are they the same)
    def imageToPixmap(self, image):
        qformat = QImage.Format_RGB888
        img = QImage(image, image.shape[1], image.shape[0] , image.strides[0], qformat)
        #img = img.rgbSwapped()  # BGR > RGB # needed for camera feed

        return QPixmap.fromImage(img)
    
    # Check def convert_cvimage)to_pixmap / def imageToPixmap (are they the same)
    def CameraToPixmap(self, image):
        qformat = QImage.Format_RGB888
        img = QImage(image, image.shape[1], image.shape[0] , image.strides[0], qformat)
        img = img.rgbSwapped()  # BGR > RGB # needed for camera feed

        return QPixmap.fromImage(img)
    
    def update_camera_feed(self):
        # This method will be called at regular intervals by the timer
        ret, frame = self.cap.read()  # read frame from webcam
        if ret:  # if frame captured successfully
            frame_inverted = cv2.flip(frame, 1)  # flip frame horizontally
            original_inverted_frame = frame_inverted.copy()  # Store the original frame

            # Add: if not self.test_started: -> update inverted_frame to corrected frame
            if self.test_calibration == True:
                    undistorted_frame = self.undistort_frame(frame, self.camera_matrix, self.camera_dist_coeff)
                    frame_inverted = undistorted_frame # cheesey replace

            if self.capture_started:
                # Call detectCorners function
                ret_corners, corners, frame_with_corners = self.detectCorners(frame_inverted, no_of_columns, no_of_rows)
                #print("Countdown is", self.countdown_seconds)

                if ret_corners and self.countdown_seconds > 0:
                    #self.update_status_signal.emit(f"Capturing in {self.countdown_seconds} seconds...")
                    print ("Capturing in", self.countdown_seconds, "seconds...")
                elif ret_corners and self.countdown_seconds == 0:
                    self.save_screenshot(original_inverted_frame)  # Have to save original Frame
                    self.countdown_seconds = 3  # Reset the countdown after saving

                if ret_corners:
                    # Display the frame with corners
                    self.pixmap = self.CameraToPixmap(frame_with_corners)
                    #self.cameraFrame.setPixmap(self.pixmap)
                else:
                    # Display the original frame
                    self.pixmap = self.CameraToPixmap(frame_inverted)
                    self.cameraFrame.setPixmap(self.pixmap)

            # Ensure the image does not scales with the label -> issue with aspect ratio TODO
            self.cameraFrame.setScaledContents(False)
            self.update()
    
    def update_countdown(self):
        if self.countdown_seconds > 0:
            self.countdown_seconds -= 1
        else:
            self.countdown_timer.stop()

            # Reset the countdown to its initial value
            self.countdown_seconds = 3
            self.countdown_timer.start()

    def detectCorners(self, image, columns, rows):
        
        # stop the iteration when specified accuracy, epsilon, is reached or specified number of iterations are completed. 
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Convert to gray for better edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners. If desired number of corners are found in the image then ret = true 
        ret, corners = cv2.findChessboardCorners(gray, (columns, rows), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS + cv2.CALIB_CB_NORMALIZE_IMAGE) # Current Falcons CamCal option

        if ret:
            print("Corners detected successfully!")
            # Refining pixel coordinates for given 2d points.
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # draw and display the chessboard corners
            cv2.drawChessboardCorners(image, (columns, rows), corners, ret)
        
            # Print the number of corners found
            print("Number of corners found:", len(corners))

        return ret, corners, image

    def save_screenshot(self, frame):
        # Ensure that the output directory exists
        output_dir = "./output"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate a timestamp for the screenshot filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

        filename = f"output/corner_{timestamp}.png"

        # Save the frame as an image
        cv2.imwrite(filename, frame)

        self.ImagesCounter += 1

        # Update the output window with the saved filename
        self.outputWindow.setText(f"Screenshot saved:\n{filename}\nTotal Images Collected: {self.ImagesCounter}")
        print(f"Screenshot saved:\n {filename}\nTotal Images Collected: {self.ImagesCounter}")

    # Check collected images with corners and start calibration if ok
    def check_done(self):
        if self.ImagesCounter < min_cap:
            rem = min_cap - self.ImagesCounter
            QMessageBox.question(self, "Warning!", f"The minimum number of captured images is set to {min_cap}.\n\nPlease collect {rem} more images", QMessageBox.Ok)
        else:
            self.timer.stop() # Stop camera feed
            self.countdown_timer.stop() # Stop countdown

            # Start Calibration
            self.perform_calibration()

            # Update button text
            self.captureButton1.setText("Capture Finished")
            self.captureButton1.setDisabled(True)

            # Emit the signal with the updated status text
            self.update_status_signal.emit("Capture Finished")

            # Update DONE button to Test Calibration
            self.doneButton1.setText("Test Calibration")
            self.doneButton1.clicked.connect(self.test_calibration) # change connect to calibartion test

    def perform_calibration(self):
        # Camera calibration to return calibration parameters (camera_matrix, distortion_coefficients)
        print("Start Calibration")

        # Emit the signal with the updated status text
        self.update_status_signal.emit("Calibration in progess...")

        # Lists to store object points and image points from all the images.
        object_points = []  # 3D points in real world space
        image_points = []   # 2D points in image plane.

        # Load saved images and find chessboard corners
        image_files = sorted(os.listdir("output")) # TODO replace output with variable
        for file_name in image_files:
            if file_name.startswith("corner_") and file_name.endswith(".png"):
                file_path = os.path.join("output", file_name)
                
                # Load the image
                frame = cv2.imread(file_path)

                # Detect corners
                ret_corners, corners, _ = self.detectCorners(frame, no_of_columns, no_of_rows)

                if ret_corners:
                    # Assuming that the chessboard is fixed on the (0, 0, 0) plane
                    object_points.append(self.generate_object_points(no_of_columns, no_of_rows, square_size))
                    image_points.append(corners)

        # check with min_cap for minimum images needed for calibration (Default is 3)
        if len(object_points) >= min_cap:
            # Generate a timestamp for the screenshot filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            # Perform camera calibration
            ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(
                object_points, image_points, (frame.shape[1], frame.shape[0]), None, None
            )

            if ret:  # if calibration was successfully
                # Display the calibration results
                self.outputWindow.setText(f"Camera matrix:{camera_matrix}")
                print("\n Camera matrix:")
                print(camera_matrix)

                # Assign camera_matrix to the instance variable
                self.camera_matrix = camera_matrix

                self.outputWindow.setText(f"Distortion coefficient:{distortion_coefficients}")
                print("\n Distortion coefficient:")
                print(distortion_coefficients)

                # Assign camera_distortion coefficient to the instance variable
                self.camera_dist_coeff = distortion_coefficients

                # Save intrinsic parameters to intrinsic.txt
                with open(f"./output/intrinsic_{timestamp}.txt", "w") as file: # TODO update output folder with variable
                    file.write("Camera Matrix:\n")
                    file.write(str(camera_matrix))
                    file.write("\n\nDistortion Coefficients:\n")
                    file.write(str(distortion_coefficients))

                self.outputWindow.setText(f"Rotation Vectors:{rvecs}")
                print("\n Rotation Vectors:")
                print(rvecs)

                self.outputWindow.setText(f"Translation Vectors:{tvecs}")
                print("\n Translation Vectors:")
                print(tvecs)

                # Save extrinsic parameters to extrinsic.txt
                with open(f"./output/extrinsic_{timestamp}.txt", "w") as file: # TODO update output folder with variable
                    for i in range(len(rvecs)):
                        file.write(f"\n\nImage {i+1}:\n")
                        file.write(f"Rotation Vector:\n{rvecs[i]}\n")
                        file.write(f"Translation Vector:\n{tvecs[i]}")
                
                self.outputWindow.setText(f"Calibration parameters saved to ./output/intrinsic_{timestamp}.txt and ./output/extrinsic_{timestamp}.txt.")
                print(f"Calibration parameters saved to ./output/intrinsic_{timestamp}.txt and ./output/extrinsic_{timestamp}.txt.")

            else:
                print("Camera Calibration failed")

    def generate_object_points(self, columns, rows, square_size):
        # Generate a grid of 3D points representing the corners of the chessboard
        objp = np.zeros((columns * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
        objp *= square_size

        return objp
    
    def test_calibration(self):
        # TODO if Pause button was pressed , camera stops !!!!
        print("Testing Calibration")

        # Emit the signal with the updated status text
        self.update_status_signal.emit("Testing in progess....")

        # Set test boolean
        self.test_started = True

        # Start update_camera_feed again
        self.timer.start(100) # Start camera feed

        if self.cal_imported == False:
            # Update DONE button to Test Calibration
            self.doneButton1.setText("Save to File")
            self.doneButton1.clicked.connect(self.save_calibration) # change connect to calibartion test
        else:
            # Update DONE button to Test Calibration
            self.doneButton1.setText("Continue to De-Warp")
            self.doneButton1.clicked.connect(self.start_pwarp) # change connect to calibartion test 
             

    def undistort_frame(self, frame, camera_matrix, distortion_coefficients):
        # Check if camera_matrix is available
        if self.camera_matrix is not None and self.camera_dist_coeff is not None:
            # Undistort the frame using the camera matrix and distortion coefficients
            undistorted_frame = cv2.undistort(frame, camera_matrix, distortion_coefficients)

            return undistorted_frame
        
        else:
            print("No camera matrix or distortion coefficient detected, showing original frame")

            return frame
        
    def save_calibration(self):
        # TODO verify why save_calibration is called when CTRL-D is pressed
        print("Saving Calibration parameters to file")

        # Emit the signal with the updated status text
        self.update_status_signal.emit("Saving in progess...")

        # Generate a timestamp for the json parameter filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Convert NumPy arrays to lists
        camera_matrix_list = self.camera_matrix.tolist() if self.camera_matrix is not None else None
        dist_coeff_list = self.camera_dist_coeff.tolist() if self.camera_dist_coeff is not None else None
        
        # Combine the data
        data = {"camera_matrix": camera_matrix_list, "dist_coeff": dist_coeff_list}
        fname = f"data_{timestamp}.json"

        #print(f"Dumping below to file {fname}: \n\n {data}")

        try:
             # Write the calibration parameters to the JSON file
            with open(f"./output/{fname}", "w") as file: # TODO update output folder with variable
                json.dump(data, file)

            # Emit the signal with the updated status text
            self.update_status_signal.emit("Calibration file Saved")
            
            # Update DONE button to Test Calibration
            self.doneButton1.setText("Continue to De-Warp")
            self.doneButton1.clicked.connect(self.start_pwarp) # change connect to calibartion test 

            # Stop Camera feed
            self.timer.stop()

        except Exception as e:
            # Handle any exceptions that may occur during file writing
            print(f"Error saving calibration file: {e}")
            self.update_status_signal.emit("Error saving calibration file")

    # Below is mostly tab2 related to Perspective-Warp
    
    # Slot function to enable or disable the "Load Image" button based on the radio button state
    def update_load_button_state(self):
        if self.radio_usb.isChecked() or self.radio_network.isChecked():
            self.loadImage.setEnabled(False)
        else:
            self.loadImage.setEnabled(True)
    
    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options
        )
        if file_name:
            # Set image Dewarp to True
            self.image_dewarp = True # TODO Does not below here

            # Load the image using OpenCv
            self.cv_image = cv2.imread(file_name)
            self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_RGB2BGR)

            # Covert to Pixmap
            pixmap = self.imageToPixmap(self.cv_image) ## Test

            # Show stuff
            self.imageFrame.setPixmap(pixmap)
            self.imageFrame.setScaledContents(False) # Set to false to prevent issues with aspect ratio

            #return self.cv_image
    
    def start_pwarp(self):       

        # Stop Camera
        self.timer.stop() # Should not be here !

        # Temp disable Start button untill all 4 points are collected TODO
        self.startButtonPwarp.setDisabled(True) # Should not be here !

        # Disable the first tab (Camera Calibration)
        self.tabs.setTabEnabled(0, False) # Should not be here !

        # Switch to the second tab (Perspective-Warp)
        self.tabs.setCurrentIndex(1) # Should not be here !

        if self.image_dewarp == True:
            print("Image Perspective Warp started")
            self.update_status_signal.emit("Image Perspective Warp started")         

            frame = self.cv_image

            # Check if 'frame' is a valid NumPy array
            if isinstance(frame, np.ndarray):

                # Disable Load image Button when import succeeded and dewarp started
                self.loadImage.setDisabled(True)  # LoadImage / load_image is confusing TODO

                if len(frame.shape) == 3:  # Check if it's a 3D array (color image)

                    self.warper_result = self.dewarp(frame) # return warper

                    # Apply camera correction if any set
                    undistorted_frame = self.undistort_frame(frame, self.camera_matrix, self.camera_dist_coeff)

                    # Perform dewarping
                    dewarped_frame = self.warper_result.warp(undistorted_frame.copy())

                    # Update the display with the dewarped image
                    self.display_dewarped_image(dewarped_frame)

                    # Print stuff and update status bar
                    print("Dewarping process completed.")
                    self.update_status_signal.emit("Dewarping process completed.")

                    # Update button text for next step
                    self.startButtonPwarp.setText("Tweak Landmarks")

                    # Need this later
                    #self.startButtonPwarp.setText("DONE")
                    #self.startButtonPwarp.clicked.connect(self.close_application) # close when done
                    
                    # TODO next steps - Tweak landmarks
                    self.startButtonPwarp.clicked.connect(self.tweak_pwarp)

                    if self.image_tuning_dewarp == False:
                        self.startButtonPwarp.clicked.disconnect(self.start_pwarp) # disconnect the previous connection

                else:
                    print("Invalid frame format: Not a 3D array (color image)")
            else:
                print("Invalid frame format: Not a NumPy array")


        elif self.usb_dewarp == True:
            #usb_dewarp()
            print("Starting usb camera de-warp") #-> update status

            # Start the camera again
            self.timer.start(100)  # Assuming 100 milliseconds per frame update (fast is 20)

            # TODO
            ret, frame = self.cap.read()  # read frame from webcam   -> use update_camera function

            if ret:  # if frame captured successfully
                #frame_inverted = cv2.flip(frame, 1)  # flip frame horizontally
                #original_inverted_frame = frame_inverted.copy()  # Store the original frame
                #undistorted_frame = self.undistort_frame(frame, self.camera_matrix, self.camera_dist_coeff)
                #dewarped_frame = self.warper.warp(undistorted_frame.copy())  # Perform dewarping
                #self.display_dewarped_image(dewarped_frame)
                pass # temp

        elif self.network_dewarp == True:
            #network_dewarp()
            print("Starting network de-warp") #-> update status

    def dewarp(self, img):

        # Add check if Tuning is started TODO
        bgimage = img.copy()
        self.display_landmarks(bgimage)

        # Added Check to see it tuning is started to either capture mouse events of arrow keys while tuning
        if self.image_tuning_dewarp == False:
            print("Landmark collection is started")

            # Start collecting landmarks with mouse clicks
            self.imageFrame.mousePressEvent = self.mouse_click_landmark_event

        if self.image_tuning_dewarp == True:
            print("Perspective Tuning is started")

            # Stop mouse press event registration
            self.imageFrame.mousePressEvent = None

            # Start collecting arrow key events --> TEST TODO Ths is not the correct way
            self.imageFrame.keyPressEvent = self.keypress_tuning_event

        # Starts a loop collecting points
        while True:
            if len(self.points) == 0:
                self.processoutputWindow.setText(f"Select the first landmark {landmark1}")
                # Draw landmark 1 on 2d field view
                # print("Drawing landmark 1:", landmark1)
                self.field_image = self.draw_landmark(self.field_image, FCS20)
                    
                # Convert to Pixman
                self.pixmap = self.imageToPixmap(self.field_image)
                pixmap = QPixmap(self.pixmap)

                #Load the image
                self.ProcessImage.setPixmap(pixmap)
                self.ProcessImage.setScaledContents(True)
                
            if len(self.points) == 1:
                self.processoutputWindow.setText(f"Select the second landmark {landmark2}")
                # Draw landmark 2 on 2d field view
                # print("Drawing landmark 2:", landmark2)
                self.field_image = self.draw_landmark(self.field_image, FCS60)
                    
                # Convert to Pixman
                self.pixmap = self.imageToPixmap(self.field_image)
                pixmap = QPixmap(self.pixmap)

                #Load the image
                self.ProcessImage.setPixmap(pixmap)
                self.ProcessImage.setScaledContents(True)
                    
            if len(self.points) == 2:
                self.processoutputWindow.setText(f"Select the third landmark {landmark3}")
                # Draw landmark 3 on 2d field view
                #print("Drawing landmark 3:", landmark3)
                self.field_image = self.draw_landmark(self.field_image, FCS02)
                    
                # Convert to Pixman
                self.pixmap = self.imageToPixmap(self.field_image)
                pixmap = QPixmap(self.pixmap)

                #Load the image
                self.ProcessImage.setPixmap(pixmap)
                self.ProcessImage.setScaledContents(True)

            if len(self.points) == 3:
                self.processoutputWindow.setText(f"Select the last landmark {landmark4}")
                # Draw landmark 4 on 2d field view
                # print("Drawing landmark 4:", landmark4)
                self.field_image = self.draw_landmark(self.field_image, FCS06)
                    
                # Convert to Pixman
                self.pixmap = self.imageToPixmap(self.field_image)
                pixmap = QPixmap(self.pixmap)

                #Load the image
                self.ProcessImage.setPixmap(pixmap)
                self.ProcessImage.setScaledContents(True)

            if len(self.points) == 4:

                # Enable Start button again when all 4 points are collected
                self.startButtonPwarp.setDisabled(False)

                if self.image_tuning_dewarp == True:
                    # tune self.point before processing them when tuning is enabled
                    print("Start tuning landmarks")
                    print(f"Tuning the selected Landmark: {self.points}")
                    self.processoutputWindow.setText(f"Tuning selected Landmarks" )

                    # TODO - need a way to update image on key event

                    self.startButtonPwarp.setText("Click when done tuning")
                    self.startButtonPwarp.clicked.connect(self.stop_tuning)

                break

            QApplication.processEvents()

        print(f"Check UI Frame | W: {self.width()}, H: {self.height()}, Supersample: {self.supersample}") # is using wrong values -> cameraFrame
        print(f"Check cameraFrame| W: {self.cameraFrame.width()}, H: {self.cameraFrame.height()}, Supersample: {self.supersample}") # is using wrong values -> cameraFrame

        # TODO is CameraFrame the best option?
        warper = Warper(points=self.points, width=self.cameraFrame.width, height=self.cameraFrame.height, supersample=self.supersample)

        return warper
    
    def stop_tuning(self):
        self.image_tuning_dewarp == False
    
    def draw_landmark(self, image, landmark):

        # TEMP Set color based on stage -> Probably not needed
        if self.image_tuning_dewarp == True:
            color = pink
        else:
            color = red

        # Draw the landmark
        cv2.circle(image, landmark, 15, (color), -1)  # Red dot for landmark
        cv2.putText(image, f"{landmark}", (landmark[0] + 20, landmark[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (lightgreen), 2, cv2.LINE_AA) # light green coords
        
        return image
    
    # Different way to display dewarped image and image landmark 

    def display_dewarped_image(self, dewarped_frame):
        # Display the dewarped image
        dewarped_pixmap = self.imageToPixmap(dewarped_frame)
        self.imageFrame.setPixmap(dewarped_pixmap)
        self.imageFrame.setScaledContents(True)

    def display_landmarks(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.imageFrame.setPixmap(pixmap)

    # To Capture landmarks with mouse clicks
    def mouse_click_landmark_event(self, event):
        if event.button() == Qt.LeftButton:
            x = event.x()
            y = event.y()

            # Print the x, y coordinates
            print(f"Mouse clicked at x: {x}, y: {y}")

            # Check if the pixmap is set on the Image Frane
            if self.imageFrame.pixmap() is not None:

                self.points.append((x, y))
                self.imageFrame.setPixmap(QPixmap())  # Clear the current pixmap
                bgimage = cv2.rectangle(self.cv_image, (x, y), (x + 2, y + 2), (green), 2)
                self.display_landmarks(bgimage)
            else:
                print("Pixmap is not set")

    def keypress_tuning_event(self, event):
        if len(self.points) == 4:  # Ensure 4 points are selected
            if event.key() == Qt.Key_Up:
                self.points[self.selected_point] = self.adjust_point(self.points[self.selected_point], 'up')
                print("Moved point up")
            elif event.key() == Qt.Key_Down:
                self.points[self.selected_point] = self.adjust_point(self.points[self.selected_point], 'down')
                print("Moved point down")
            elif event.key() == Qt.Key_Left:
                self.points[self.selected_point] = self.adjust_point(self.points[self.selected_point], 'left')
                print("Moved point left")
            elif event.key() == Qt.Key_Right:
                self.points[self.selected_point] = self.adjust_point(self.points[self.selected_point], 'right')
                print("Moved point right")
            #elif event.key() == Qt.Key_Tab:

                # TODO Tab does not stay in the frame                
            #    print(f"Before Tab update: {self.selected_point}")

            #    self.selected_point = (self.selected_point + 1) % 4

            #    print(f"After Tab update: {self.selected_point}")
            #    print("Switched to the next point")

            #    event.accept()  # Accept the event to prevent default tab 
            elif event.key() == Qt.Key_1:
                self.selected_point = 0
                print("Selected landmark 1")
            elif event.key() == Qt.Key_2:
                self.selected_point = 1
                print("Selected landmark 2")
            elif event.key() == Qt.Key_3:
                self.selected_point = 2
                print("Selected landmark 3")
            elif event.key() == Qt.Key_4:
                self.selected_point = 3
                print("Selected landmark 4")

            else:
                super().keyPressEvent(event)  # Pass other key events to the base class

            # Make rectangle red when selected
            print(f"Updating perspective transform, using: self.points (pts{self.selected_point}): {self.points}")

            # Check if the pixmap is set on the Image Frane
            if self.imageFrame.pixmap() is not None:

                # TODO update selected point -> update perspective warp
                print("Update Image Frame / PWarp view")

                # Perform pwarp at every key press
                self.warper_result = self.dewarp(self.cv_image) # return warper

                # Apply camera correction if any set
                undistorted_frame = self.undistort_frame(self.cv_image, self.camera_matrix, self.camera_dist_coeff)

                # Perform dewarping
                dewarped_frame = self.warper_result.warp(undistorted_frame.copy())

                # Update the display with the dewarped image
                self.display_dewarped_image(dewarped_frame)
                    
            else:
                print("Pixmap is not set")

    def adjust_point(self, point, direction):
        """ Adjust the point based on arrow key input """
        x, y = point
        if direction == 'up':
            return (x, y - 1)
        elif direction == 'down':
            return (x, y + 1)
        elif direction == 'left':
            return (x - 1, y)
        elif direction == 'right':
            return (x + 1, y)
        return point
    
    # do i need this or can i use display_dewarped_image again
    def update_perspective_transform(self, pts1, pts2):
        
        self.frame = self.cv_image

         # Call the height and width method to get the actual value of the frame
        W = self.cameraFrame.width() # is it cheating to use frame instead of img?
        H = self.cameraFrame.height()

        # Update and display the perspective-transformed image
        self.pts1 = np.float32(pts1)
        self.pts2 = np.float32(pts2)

        M = cv2.getPerspectiveTransform(self.pts1, self.pts2)

        supersample = self.supersample

        if self.frame is None:
            self.frame = cv2.warpPerspective(self.frame,M,(W*supersample,H*supersample))
        else:
            self.frame[:] = cv2.warpPerspective(self.frame,M,(W*supersample,H*supersample))  # -> issue using from frame.shape-
        
        # Convert the adjusted image to QPixmap and display it
        self.display_landmarks(self.frame) # TODO


    def tweak_pwarp(self):

        #frame = self.cv_image
        # Print new instruction
        self.processoutputWindow.setText("Tuning Landmarks started")

        # Add check if Tuning is started TODO
        self.image_tuning_dewarp = True # Track if an image tuning is used for dewarping

        # Start de warp again for tuning
        self.start_pwarp()

    def close_application(self):
        QApplication.quit()
            

class CamCalMain(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "Falcons De-Warp Tool - BETA"
        self.setWindowTitle(self.title)
        self.setGeometry(0, 0, 800, 600) # #self.setGeometry(self.left, self.top, self.width, self.height)

        self.check_output_empty() # check output folder
        self.init_ui() # initialize UI

    def init_ui(self):
        # Create the central widget
        self.camera_widget = CameraWidget(self)
        self.setCentralWidget(self.camera_widget)

        # Create the menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        # Add actions to the File menu to import calibration JSON
        import_calibration = QAction("Import Calibration", self) # Load caibration file?
        import_calibration.triggered.connect(self.load_calibration)
        file_menu.addAction(import_calibration)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Ctrl+D")
        file_menu.addAction(exit_action)

        statusbar = QStatusBar()
        self.setStatusBar(statusbar)
        statusbar.showMessage("Status: Not started")

        # Connect the signal to the slot for updating the status bar
        self.camera_widget.update_status_signal.connect(self.update_status_bar)

    def check_output_empty(self):
        # Check if the output folder exists
        output_folder = "./output"
        if not os.path.exists(output_folder):
            QMessageBox.information(self, "Info", "The output folder does not exist.", QMessageBox.Ok)

            return

        # Check if there are existing images in the output folder
        existing_images = [f for f in os.listdir(output_folder) if f.startswith("corner_") and f.endswith(".png")]

        if existing_images:
            # Ask the user if they want to delete existing images
            reply = QMessageBox.question(
                self, "Existing Images", "There are existing images in the output folder. Do you want to delete them?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                # Delete existing images
                for image_file in existing_images:
                    file_path = os.path.join(output_folder, image_file)
                    os.remove(file_path)
                
                # Inform the user about the deletion
                QMessageBox.information(self, "Deletion Complete", "Existing images have been deleted.", QMessageBox.Ok)

            else:
                # If the user chooses not to delete, inform them and exit the method
                QMessageBox.information(self, "Calibration Canceled", "Calibration process canceled.", QMessageBox.Ok)

                return

    def load_calibration(self, filename):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        # TODO open ouput folder to load JSON
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Calibration File", "", "JSON Files (*.json);;All Files (*)", options=options 
        )

        if file_name:
            try:
                with open(file_name, "r") as f:
                    data = json.load(f)
                    self.camera_matrix = data.get("camera_matrix")
                    self.camera_dist_coeff = data.get("dist_coeff")
                    
                    # Emit the signal with the updated status text
                    self.camera_widget.update_status_signal.emit(f"Calibration parameters loaded from {file_name}")
                    print(f"Calibration parameters loaded from {file_name}")

                    # Set tracker to True in camera_widget, needed in test_calibration
                    self.camera_widget.cal_imported = True

                    # Update button text
                    self.camera_widget.captureButton1.setText("Capture Finished")
                    # Disable Capture Button when import succeeded
                    self.camera_widget.captureButton1.setDisabled(True)

                    # Cheesy set ImageCOunter to minimum to start testing
                    self.ImagesCounter = min_cap
                    self.camera_widget.ImagesCounter = min_cap

                    # Start Camera
                    self.camera_widget.start_capture()

                    # Update DONE button to Test Calibration
                    self.camera_widget.doneButton1.setText("Test Calibration")
                    self.camera_widget.doneButton1.clicked.connect(self.camera_widget.test_calibration) # change connect to calibartion test

            except FileNotFoundError:
                print(f"File {filename} not found.")
                self.camera_widget.update_status_signal.emit(f"File not found: {file_name}")

            except Exception as e:
                print(f"Error loading calibration file: {e}")
                self.camera_widget.update_status_signal.emit("Error loading calibration file")

    def update_status_bar(self, status_text):
        # Update the status bar text
        self.statusBar().showMessage(status_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = CamCalMain()
    ex.show()
    sys.exit(app.exec_())
