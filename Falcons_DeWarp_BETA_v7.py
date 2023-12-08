#!/usr/bin/env python3

import sys
import cv2
import time
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
from PyQt5.QtGui import QIcon, QPixmap, QImage, QColor
from PyQt5.QtCore import pyqtSlot, pyqtSignal, Qt, QTimer

# Dependency OpenCV 4.x, PyQT (Default in UBNT 22.04)
# TODO Default disable tab2, only enable after finish Calibartion or loading Calibration JSON

#Config Default Variables - Enter their values according to your Checkerboard, normal 64 (8x8) -1 inner corners only
no_of_columns = 7  #number of columns of your Checkerboard
no_of_rows = 7  #number of rows of your Checkerboard
square_size = 27.0 # size of square on the Checkerboard in mm
min_cap = 3 # minimum or images to be collected by capturing (Default is 10), minimum is 3

# Assuming the soccer field is 22 x 14 meters
soccer_field_width = 22
soccer_field_length = 14

################# Perspective Warp Classes ################# 

class Warper(object):
    def __init__(self,points,width=640,height=480,supersample=2,interpolation=None):
        self.points = points
        self.width  = width
        self.height = height
        self.supersample = supersample

        # TODO use the correct landmarks 2,0 6,0 0,2 0,6 
        self.pts1 = np.float32([points[0],points[1],points[3],points[2]])

        print("self.pts1:", self.pts1)

        # Call the height and width method to get the actual value of the frame
        W = self.width()
        H = self.height()

        print(f"Check 1 | W: {W}, H: {H}, Supersample: {supersample}") # is using wrong values -> cameraFrame

        ## TEST
        # Check if W and H are integers or floats
        if not isinstance(W, (int, float)) or not isinstance(H, (int, float)):
            raise ValueError("Width and height should be numerical values.")

        # Check if supersample is a numerical value
        if not isinstance(supersample, (int, float)):
            raise ValueError("Supersample should be a numerical value.")

        self.pts2 = np.float32([[0,0],[W*supersample,0],[0,H*supersample],[W*supersample,H*supersample]])

        print("self.pts2:", self.pts2) # Now set to outer display coordinates TODO

        self.M = cv2.getPerspectiveTransform(self.pts1,self.pts2)
        self.dst = None
        if (interpolation == None):
            self.interpolation = cv2.INTER_CUBIC
        else:
            self.interpolation = interpolation

    def warp(self,img,out=None):
        # Call the height and width method to get the actual value of the frame
        W = self.width()  # Call the width method to get the actual value
        H = self.height()  # Call the height method to get the actual value

        M = self.M
        supersample = self.supersample

        print(f"Check 2 | W: {W}, H: {H}, Supersample: {supersample}") # is using wrong values -> cameraFrame

        if self.dst is None:
            self.dst = cv2.warpPerspective(img,M,(W*supersample,H*supersample))
        else:
            self.dst[:] = cv2.warpPerspective(img,M,(W*supersample,H*supersample))

        # unnecessarily complicated
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
    countdown_seconds = 2 # 3 seconds feels long

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

        # Need the camera object in this Widget
        self.cap = cv2.VideoCapture(0) # webcam object
        self.pixmap = None

        # Add cv_image
        self.cv_image = None

        # Placeholder to count the ammount of images saved
        self.ImagesCounter = 0

        # Initialize camera_matrix as None
        self.camera_matrix = None

        # Initialize camera_distortion coefficients as None
        self.camera_dist_coeff = None

        # Do i need rvecs and tvecs?

        # Define basic colors for Soccer Field (RGB)
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.gray = (128, 128, 128)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.lightgreen = (144, 238, 144)
        self.blue = (0, 0, 255)
        self.lightblue = (173, 216, 230)
        self.pink = (255, 192, 203)
        self.magenta = (255, 0, 255)

        # Define basic line thickness to draw soccer field
        self.line_thickness = 2

        # Set variable to store selected objectpoint for dewarp
        self.points = []

         # Add the supersample attribute
        self.supersample = 2 

        #..:: Start UI layout ::..#

        # Use a central layout for the entire widget
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
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
        #self.outputWindow = QLineEdit()
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
        #self.tab2.layout = QVBoxLayout(self.tab2)

        # Create Vertical Box Layout for tab1 inner frame
        self.tab2inner = QWidget()
        self.tab2inner.layout = QVBoxLayout(self.tab2inner)

        # Add Load Image on top (For now load from R8 folder)
        self.loadImage = QPushButton("Load Image", self.tab2inner)
        self.loadImage.clicked.connect(self.load_image)

        self.tab2inner.layout.addWidget(self.loadImage)

        # Add Image Frame 
        self.imageFrame = QLabel("Image will be displayed here", self.tab2inner)
        #self.imageFrame.mousePressEvent = self.getPos # Assign method not calling it with ()

        # Set word wrap to ensure text fits within the QLabel width
        self.imageFrame.setWordWrap(True)

        # Set alignment to center the text within the QLabel
        self.imageFrame.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.imageFrame.setFrameShape(QFrame.Box)
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
        # self.ProcessFrame.setFixedWidth(450)
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
        #label_height = 240 # To small when using color
        label_height = 300

        # Draw and display soccer field in the ProcessFrame
        soccer_field_image = self.draw_soccer_field()
        # soccer_field_pixmap = self.convert_cvimage_to_pixmap(soccer_field_image)
        soccer_field_pixmap = self.imageToPixmap(soccer_field_image) ## Test

        if soccer_field_pixmap:
             # Load the image using QPixmap
            pixmap = QPixmap(soccer_field_pixmap)

            # Load the image
            #pixmap = cv2.imread(file_name)
            self.ProcessImage.setPixmap(pixmap)
            self.ProcessImage.setScaledContents(True) # needed or field wont fit

        # Calculate the corresponding width based on the aspect ratio
        label_width = int(label_height * aspect_ratio)

        # Set size policy and fixed size
        self.ProcessImage.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ProcessImage.setFixedSize(label_width, label_height)

        self.ProcessFrame.layout.addWidget(self.ProcessImage)   

        # Add Process Options TODO

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
            self.timer.start(100)  # Set the interval in milliseconds (e.g., 100 milliseconds)
            
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
        # img = img.rgbSwapped()  # BGR > RGB # remove it here for now
        # print("Pixmap conversion successful")
        return QPixmap.fromImage(img)
    
    # Check def convert_cvimage)to_pixmap / def imageToPixmap (are they the same) -> This one not used now
    #def convert_cvimage_to_pixmap(self, image):
    #    height, width, channel = image.shape
    #    bytesPerLine = 3 * width
    #    qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
    #    return QPixmap.fromImage(qImg)
    
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
                    self.pixmap = self.imageToPixmap(frame_with_corners)
                    self.cameraFrame.setPixmap(self.pixmap)
                else:
                    # Display the original frame
                    self.pixmap = self.imageToPixmap(frame_inverted)
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
        self.timer.start(100) # Start camera feed (is based on timer)

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

            # Maybe via dialog box (TODO)

    # Below is mostly tab2 related to Perspective-Warp

    # Add logic to select usb, image or network TODO
    
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
            # Load the image using QPixmap
            # pixmap = QPixmap(file_name)

            # Set image Dewarp to True
            self.image_dewarp = True

            # Load the image using OpenCv
            self.cv_image = cv2.imread(file_name)
            self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_RGB2BGR)

            # Covert to Pixmap
            # pixmap = self.convert_cvimage_to_pixmap(self.cv_image)
            pixmap = self.imageToPixmap(self.cv_image) ## Test

            # Show stuff
            self.imageFrame.setPixmap(pixmap)
            self.imageFrame.setScaledContents(True)

            #return self.cv_image
    
    def draw_soccer_field(self):
        # Below is by no means official sizing for MSL
        height, width = 500, 1000
        # Create a blank image (blank background)
        # image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Create a blank image (light green background)
        image = np.ones((height, width, 3), dtype=np.uint8) * np.array(self.lightgreen, dtype=np.uint8)

        # Set line color
        line_color = self.white

        # Draw the soccer field boundary
        # The rectangle goes from (50, 50) to (950, 450) on the picture,
        cv2.rectangle(image, (50, 50), (950, 450), line_color, self.line_thickness)

        # Draw the center circle
        # The circle is centered at (500, 250), has a radius of 40 pixels
        # PLease note that 500, 250 is 0, 0 in Field Coordinate System 
        cv2.circle(image, (500, 250), 60, line_color, self.line_thickness)

        # Draw goal areas
        # The rectangle goes from (xx, xx) to (xxx, xxx) on the picture,
        cv2.rectangle(image, (50, 200), (100, 300), line_color, self.line_thickness)
        cv2.rectangle(image, (900, 200), (950, 300), line_color, self.line_thickness)

        # Draw goals
        # The rectangle goes from (xx, xx) to (xxx, xxx) on the picture,
        cv2.rectangle(image, (50, 225), (51, 275), line_color, self.line_thickness)
        cv2.rectangle(image, (949, 225), (950, 275), line_color, self.line_thickness)

        # Draw penalty areas
        # The rectangle goes from (xx, xx) to (xxx, xxx) on the picture,
        cv2.rectangle(image, (50, 100), (200, 400), line_color, self.line_thickness)
        cv2.rectangle(image, (800, 100), (950, 400), line_color, self.line_thickness)

        # Draw the middle line
        # The line goes from (500, 50) to (500, 450)
        cv2.line(image, (500, 50), (500, 450), line_color, self.line_thickness)

        # Red dot at position (500, 250) with a radius of 5
        # 500, 250 = 0, 0 in FCS (camera position)
        cv2.circle(image, (500, 250), 5, (self.red), -1)  # -1 means fill the circle

        # 500, 450 = 6, 0 in FCS is landmark for top left 1/4 of the field
        cv2.circle(image, (500, 450), 5, (self.gray), -1)

        # 500, 325 = 2, 0 in FCS if we calculate it
        cv2.circle(image, (500, 325), 5, (self.gray), -1)

        return image
    
    def start_pwarp(self):       

        # Stop Camera
        self.timer.stop()

        # Emit the signal with the updated status text
        print("Starting Perspective-Warp")
        self.update_status_signal.emit("Perspective Warp started")

        # Disable the first tab (Camera Calibration)
        self.tabs.setTabEnabled(0, False)

        # Switch to the second tab (Perspective-Warp)
        self.tabs.setCurrentIndex(1)

        #####################################
        #   Next Piece of great code ...... #
        #####################################

        # For Testing 
        # 1. Load image from R8 folder (1 camera view of 1/4 of the field)
        #    Camera on spot in the Center Cirle (FCS 0,0) , center of the camera pointing to one corner 
        # 2. Show Image and a Birds eye view of the Soccer field below with corresponding land marks to select 
        # 3. Select 4 coordinates on Image that correspond to 4 landmarks on the birdseye view soccer field
        #    2.0 (just outside the Center Cirle towards the side line)
        #    6.0 ( Where the middle line meets the side line)
        #    0.2 (just outside the Center Circle towards the goal)
        #    0.6 (Penalty Spot)
        # 4. Perform Perspective Warp and show output plotted on the Birdseye view.
        # 5. Optimize landmark selection selections
        # 6. Save to binairy cv mat file

        if self.image_dewarp == True:
            print("Starting image de-warp") #-> update status

            # Use the loaded image directly
            # pixmap = self.imageFrame.pixmap()

            # image = pixmap.toImage()
            # cv_frame = np.array(image.rgbSwapped())  # Convert to NumPy array

            frame = self.cv_image

            # Check if 'frame' is a valid NumPy array
            if isinstance(frame, np.ndarray):
                if len(frame.shape) == 3:  # Check if it's a 3D array (color image)

                    #self.dewarp(frame) # return warper
                    self.warper_result = self.dewarp(frame) # return warper

                    #original_inverted_frame = frame_inverted.copy()  # Store the original frame
                    undistorted_frame = self.undistort_frame(frame, self.camera_matrix, self.camera_dist_coeff)
                    dewarped_frame = self.warper_result.warp(undistorted_frame.copy())  # Perform dewarping

                    # Update the display with the dewarped image
                    self.display_dewarped_image(dewarped_frame)
    
                    # Emit the signal with the updated status text
                    QMessageBox.information(self, "Dewarping Complete", "Dewarping process completed.", QMessageBox.Ok)
                    print("Dewarping process completed.")
                    self.update_status_signal.emit("Dewarping process completed.")

                    # Update button text
                    self.startButtonPwarp.setText("DONE")

                    # Disable Capture Button when import succeeded
                    self.loadImage.setDisabled(True)  # LoadImage / load_image confusing

                    # TODO next steps

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

            # TODO


    def dewarp(self, img):
        bgimage = img.copy()
        self.display_image(bgimage)
        self.imageFrame.mousePressEvent = self.mouse_click_event

        while True:
            if len(self.points) == 4:
                break
            QApplication.processEvents()

        warper = Warper(points=self.points, width=self.width, height=self.height, supersample=self.supersample)

        return warper

    def display_dewarped_image(self, dewarped_frame):
        # Display the dewarped image
        dewarped_pixmap = self.imageToPixmap(dewarped_frame)
        self.imageFrame.setPixmap(dewarped_pixmap)
        self.imageFrame.setScaledContents(True)
        QMessageBox.information(self, "Dewarping Complete", "Dewarping process completed.", QMessageBox.Ok)

    def display_image(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.imageFrame.setPixmap(pixmap)

    def mouse_click_event(self, event):
        if event.button() == Qt.LeftButton:
            x = event.x()
            y = event.y()

            # Print the x, y coordinates
            print(f"Mouse clicked at x: {x}, y: {y}")

            # Check if the pixmap is set on the Image Frane
            if self.imageFrame.pixmap() is not None:

                self.points.append((x, y))
                self.imageFrame.setPixmap(QPixmap())  # Clear the current pixmap

                #bgimage = cv2.cvtColor(self.imageFrame.pixmap().toImage().bits(), cv2.COLOR_RGBA2RGB)
                
                #cv2.rectangle(bgimage, (x, y), (x + 2, y + 2), (0, 255, 0), 2)
                bgimage = cv2.rectangle(self.cv_image, (x, y), (x + 2, y + 2), (0, 255, 0), 2)
                self.display_image(bgimage)
            else:
                print("Pixmap is not set")
            

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
