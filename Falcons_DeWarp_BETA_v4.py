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
)
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import pyqtSlot, pyqtSignal, Qt, QTimer

# https://www.pythontutorial.net/pyqt/pyqt-qmainwindow/
# https://learnopencv.com/camera-calibration-using-opencv/
# https://www.youtube.com/watch?v=p1kCR1i2nF0

# Dependency OpenCV 4.x and MatPlotLib
# TODO Adding matplotlib to draw Field / Better visualization of results -> or other ...

#Config Default Variables - Enter their values according to your Checkerboard, normal 64 (8x8) -1 inner corners only
no_of_columns = 7  #number of columns of your Checkerboard
no_of_rows = 7  #number of rows of your Checkerboard
# square_size = 37.0 # size of square on the Checkerboard in mm
square_size = 27.0 # size of square on the Checkerboard in mm
min_cap = 3 # minimum or images to be collected by capturing (Default is 10), minimum is 3

#class MyTabsWidget(QWidget):
class CameraWidget (QWidget):
    # Add a signal for updating the status
    update_status_signal = pyqtSignal(str)

    # Set the initial countdown time to save images
    countdown_seconds = 3

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

        #Not used: 
        #self.corner_detection_started = False
        #self.calibration_started = False

        # Need the camera object in this Widget
        self.cap = cv2.VideoCapture(0) # webcam object
        self.pixmap = None

        # Placeholder to count the ammount of images saved
        self.ImagesCounter = 0

        # Initialize camera_matrix as None
        self.camera_matrix = None

        # Initialize camera_distortion coefficients as None
        self.camera_dist_coeff = None

        # Do i need rvecs and tvecs?

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

        ####################
        # Create first tab #
        ####################

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
        self.imageLabel = QLabel("Image will be displayed here", self.tab1inner)
        self.imageLabel.setAlignment(Qt.AlignCenter|Qt.AlignVCenter)
        self.imageLabel.resize(640, 480)
        self.imageLabel.setFrameShape(QFrame.Box)
        self.tab1inner.layout.addWidget(self.imageLabel)

        # Add tab1inner to tab1
        #self.tab1.layout.addWidget(self.tab1inner)

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
        option_label = QLabel("Options:")
        self.optionsFrame.layout.addWidget(option_label)

        # Add columns option
        self.columnsLabel = QLabel('# of Columns:', self)
        #self.columnsLabel.resize(220, 30)
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

        # Add output Display (display at start to small)
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

        #####################
        # Create second tab #
        #####################

        self.tab2.layout = QVBoxLayout(self.tab2)

        self.pushButton2 = QPushButton("Load Image", self.tab2)
        self.pushButton2.clicked.connect(self.load_image)
        self.tab2.layout.addWidget(self.pushButton2)

        self.image_label = QLabel(self.tab2)
        self.tab2.layout.addWidget(self.image_label)

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
    
    def imageToPixmap(self, image):
        qformat = QImage.Format_RGB888
        img = QImage(image, image.shape[1], image.shape[0] , image.strides[0], qformat)
        img = img.rgbSwapped()  # BGR > RGB
        # print("Pixmap conversion successful")
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
                    self.pixmap = self.imageToPixmap(frame_with_corners)
                    self.imageLabel.setPixmap(self.pixmap)
                else:
                    # Display the original frame
                    self.pixmap = self.imageToPixmap(frame_inverted)
                    self.imageLabel.setPixmap(self.pixmap) # Update the QLabel with the captured image

            # Ensure the image scales with the label
            self.imageLabel.setScaledContents(True)
            self.update()
    
    def update_camera_feed_defect(self):
        # This method will be called at regular intervals by the timer
        ret, frame = self.cap.read()  # read frame from webcam
        if ret:  # if frame captured successfully
            frame_inverted = cv2.flip(frame, 1)  # flip frame horizontally
            original_inverted_frame = frame_inverted.copy()  # Store the original frame

            # Add: if not self.test_started: -> update inverted_frame to corrected frame
            # if not self.test_calibration:
            if self.test_calibration == False:
                print("Self test is False")
                if self.capture_started:
                    print("Self capture is True")
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
                    self.imageLabel.setPixmap(self.pixmap)

            else:
                #if self.test_calibration:
                if self.test_calibration == True:
                    print("Self test is True")
                    # undistort_frame(self, frame, camera_matrix, distortion_coefficients):
                    undistorted_frame = self.undistort_frame(frame, self.camera_matrix, self.camera_dist_coeff)
                    frame_inverted = undistorted_frame # cheesey replace
                else:
                    print("Self test is False and calibration is False")
                    # Display the original frame
                    self.pixmap = self.imageToPixmap(frame_inverted)
                    self.imageLabel.setPixmap(self.pixmap) # Update the QLabel with the captured image

            # Ensure the image scales with the label
            self.imageLabel.setScaledContents(True)
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
        # Only detect Corners when testing is False

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

        # TODO Cleanup at start or end

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
        #else:
        #    print("Not enough images found for calibration. Make sure the images are stored in the output folder.")

    def generate_object_points(self, columns, rows, square_size):
        # Generate a grid of 3D points representing the corners of the chessboard
        objp = np.zeros((columns * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
        objp *= square_size
        return objp
    
    def test_calibration(self): # need: camera_matrix, distortion_coefficients
        # TODO if Pause butotn was pressed , camera is stopped doing testing !!!!
        print("Testing Calibration")

        # Emit the signal with the updated status text
        self.update_status_signal.emit("Testing in progess....")
        
        # source: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

        # Set test boolean
        self.test_started = True

        # Start update_camera_feed again
        self.timer.start(100) # Start camera feed (is based on timer)
        # self.update_camera_feed()clear

        #print (f"File imported is {self.cal_imported}")

        if self.cal_imported == False:
            # Update DONE button to Test Calibration
            self.doneButton1.setText("Save to File")
            self.doneButton1.clicked.connect(self.save_calibration) # change connect to calibartion test
        else:
            # Update DONE button to Test Calibration
            self.doneButton1.setText("Continue to De-Warp")
            self.doneButton1.clicked.connect(self.start_dewarp) # change connect to calibartion test 
             

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
        
        # Example
        #camera_matrix=[[ 532.80990646 ,0.0,342.49522219],[0.0,532.93344713,233.88792491],[0.0,0.0,1.0]]
        #dist_coeff = [-2.81325798e-01,2.91150014e-02,1.21234399e-03,-1.40823665e-04,1.54861424e-01]
        #data = {"camera_matrix": camera_matrix, "dist_coeff": dist_coeff}

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

            #self.start_dewarp()
            
            # Update DONE button to Test Calibration
            self.doneButton1.setText("Continue to De-Warp")
            self.doneButton1.clicked.connect(self.start_dewarp) # change connect to calibartion test 

            # Stop Camera feed
            self.timer.stop()

        except Exception as e:
            # Handle any exceptions that may occur during file writing
            print(f"Error saving calibration file: {e}")
            self.update_status_signal.emit("Error saving calibration file")

            # Maybe via dialog box (TODO)

    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options
        )
        if file_name:
             # Load the image using QPixmap
            pixmap = QPixmap(file_name)

            # Load the image
            #pixmap = cv2.imread(file_name)

            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)

    def start_dewarp(self):
        print("Starting De-Warp")

        # Stop Camera
        self.timer.stop()

        # Emit the signal with the updated status text
        self.update_status_signal.emit("De-warp started")

        # Disable the first tab (Camera Calibration)
        self.tabs.setTabEnabled(0, False)

        # Switch to the second tab (Perspective-Warp)
        self.tabs.setCurrentIndex(1)

        #####################################
        #   Next Piece of great code ...... #
        #####################################

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

        # Add actions to the File menu -> Do i need this? TODO
        open_action = QAction("Open", self) # Load caibration file?
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

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

        #self.show()

    def check_output_empty(self):
        # Check if there are existing images in the output folder
        existing_images = [f for f in os.listdir("./output") if f.startswith("corner_") and f.endswith(".png")]

        if existing_images:
            # Ask the user if they want to delete existing images
            reply = QMessageBox.question(
                self, "Existing Images", "There are existing images in the output folder. Do you want to delete them?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                # Delete existing images
                for image_file in existing_images:
                    file_path = os.path.join("./output", image_file)
                    os.remove(file_path)
                
                # Inform the user about the deletion
                QMessageBox.information(self, "Deletion Complete", "Existing images have been deleted.", QMessageBox.Ok)

            else:
                # If the user chooses not to delete, inform them and exit the method
                QMessageBox.information(self, "Calibration Canceled", "Calibration process canceled.", QMessageBox.Ok)
                return

    def open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options
        )
        if file_name:
            self.camera_widget.load_image(file_name)

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

                    #############################################################################

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

                    #############################################################################

            except FileNotFoundError:
                print(f"File {filename} not found.")
                # Below only available in tab1
                #self.update_status_signal.emit(f"File not found: {file_name}")
                self.camera_widget.update_status_signal.emit(f"File not found: {file_name}")
            except Exception as e:
                print(f"Error loading calibration file: {e}")
                # Below onlu available in tab1
                #self.update_status_signal.emit("Error loading calibration file")
                self.camera_widget.update_status_signal.emit("Error loading calibration file")

    def update_status_bar(self, status_text):
        # Update the status bar text
        self.statusBar().showMessage(status_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = CamCalMain()
    ex.show()
    sys.exit(app.exec_())
