import datetime
import json
import os

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .common import MarkerColors, SoccerFieldColors
from .config import get_config
from .soccer_field import SoccerField
from .warper import Warper


class CameraWidget(QWidget):
    # Add a signal for updating the status
    update_status_signal = pyqtSignal(str)

    # Set the initial countdown time to save images
    countdown_seconds = 3  # 3 seconds feels long

    def __init__(self, parent: QMainWindow):
        super(QWidget, self).__init__(parent)

        self.config = get_config()
        self.min_cap = 3

        # Add a QTimer countdown timer
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)

        # Add a QTimer for continuous camera feed updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_feed)

        # Add a flag to track state
        self.capture_started = False  # Track if caputure is started
        self.test_started = False  # Track if test is started
        self.cal_imported = False  # Track if imported calibration is used
        self.image_dewarp = False  # Track if an image is used for dewarping
        self.usb_dewarp = False  # Track if an USB camera is used for dewarping
        self.network_dewarp = False  # Track if an network stream is used for dewarping
        self.image_tuning_dewarp = False  # Track if an image tuning is used for dewarping

        # Need the camera object in this Widget
        self.cap = cv2.VideoCapture(0)  # webcam object
        self.pixmap = None

        # Add cv_image Temp (distorted frame)
        self.cv_image = None

        # Add undistorted frame
        self.undistorted_frame = None

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

        # self.landmark_points = []
        # Also part of Warper Class
        self.landmark_points = np.array(
            [
                self.config.field_coordinates_lm1,
                self.config.field_coordinates_lm2,
                self.config.field_coordinates_lm4,
                self.config.field_coordinates_lm3,
            ]
        )
        # np.float32([FCS20, FCS60, FCS06, FCS02])  # Odd order ofcourse but looks ok TODO

        # Add the supersample attribute
        self.supersample = 2

        # ..:: Start UI layout ::..#

        # Use a central layout for the entire widget
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        # self.tabs.setFocusPolicy(Qt.ClickFocus)  # or Qt.StrongFocus
        self.tabs.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # TEST TODO

        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab2.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # TEST TODO
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
        # self.optionsFrame.layout.addStretch(1)

        # Set fixed width for optionsFrame
        self.optionsFrame.setFixedWidth(400)

        # Add options widgets to optionsFrame:
        option_label = QLabel("Chessboard Options:")
        self.optionsFrame.layout.addWidget(option_label)

        # Add columns option
        self.columnsLabel = QLabel("# of Columns:", self)
        # self.columnsLabel.resize(220, 40)
        self.no_of_columns = self.config.no_of_columns
        self.columnsInput = QLineEdit(str(self.no_of_columns), self)
        self.optionsFrame.layout.addWidget(self.columnsLabel)
        self.optionsFrame.layout.addWidget(self.columnsInput)

        # Add row option
        self.rowsLabel = QLabel("# of Rows:", self)
        self.no_of_rows = self.config.no_of_rows
        self.rowsInput = QLineEdit(str(self.no_of_rows), self)
        self.optionsFrame.layout.addWidget(self.rowsLabel)
        self.optionsFrame.layout.addWidget(self.rowsInput)

        # Add square Size option
        self.squareSizeLabel = QLabel("Square size (mm):", self)
        self.square_size = self.config.square_size
        self.squareSizeRow = QLineEdit(str(self.square_size), self)
        self.optionsFrame.layout.addWidget(self.squareSizeLabel)
        self.optionsFrame.layout.addWidget(self.squareSizeRow)

        # Add output Display (display at start is to small)
        self.outputWindowLabel = QLabel("Output Display:", self)
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
        # self.ProcessFrame.layout.addStretch(1)

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
        aspect_ratio = self.config.soccer_field_width / self.config.soccer_field_length

        # Set the height of the QLabel to fixed pixels
        label_height = 300

        # Draw and display soccer field in the ProcessFrame
        field_drawer = SoccerField(self.config)
        soccer_field_image = field_drawer.generate_field()

        # TEMP store it
        self.field_image = soccer_field_image

        # Convert to Pixmap
        soccer_field_pixmap = self.imageToPixmap(soccer_field_image)

        if soccer_field_pixmap:
            # Load the image using QPixmap
            pixmap = QPixmap(soccer_field_pixmap)
            self.ProcessImage.setPixmap(pixmap)
            self.ProcessImage.setScaledContents(True)  # needed or field wont fit

        # Calculate the corresponding width based on the aspect ratio
        label_width = int(label_height * aspect_ratio)

        # Set size policy and fixed size
        self.ProcessImage.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ProcessImage.setFixedSize(label_width, label_height)

        self.ProcessFrame.layout.addWidget(self.ProcessImage)

        # Add process Output Display (display at start is to small)
        self.processOutputWindowLabel = QLabel("Output Display:", self)
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
        if not self.capture_started:
            # Read user-input values for columns, rows, and square size
            self.no_of_columns = int(self.columnsInput.text())
            self.no_of_rows = int(self.rowsInput.text())
            self.square_size = float(self.squareSizeRow.text())

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
        img = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        # img = img.rgbSwapped()  # BGR > RGB # needed for camera feed

        return QPixmap.fromImage(img)

    # Check def convert_cvimage)to_pixmap / def imageToPixmap (are they the same)
    def CameraToPixmap(self, image):
        qformat = QImage.Format_RGB888
        img = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
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
                frame_inverted = undistorted_frame  # cheesey replace

            if self.capture_started:
                # Call detectCorners function
                ret_corners, corners, frame_with_corners = self.detectCorners(
                    frame_inverted, self.no_of_columns, self.no_of_rows
                )
                # print("Countdown is", self.countdown_seconds)

                if ret_corners and self.countdown_seconds > 0:
                    # self.update_status_signal.emit(f"Capturing in {self.countdown_seconds} seconds...")
                    print("Capturing in", self.countdown_seconds, "seconds...")
                elif ret_corners and self.countdown_seconds == 0:
                    self.save_screenshot(original_inverted_frame)  # Have to save original Frame
                    self.countdown_seconds = 3  # Reset the countdown after saving

                if ret_corners:
                    # Display the frame with corners
                    self.pixmap = self.CameraToPixmap(frame_with_corners)
                    # self.cameraFrame.setPixmap(self.pixmap)
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
        ret, corners = cv2.findChessboardCorners(
            gray,
            (columns, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )  # Current Falcons CamCal option

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
        if self.ImagesCounter < self.min_cap:
            rem = self.min_cap - self.ImagesCounter
            QMessageBox.question(
                self,
                "Warning!",
                f"The minimum number of captured images is set to {self.min_cap}.\n\nPlease collect {rem} more images",
                QMessageBox.Ok,
            )
        else:
            self.timer.stop()  # Stop camera feed
            self.countdown_timer.stop()  # Stop countdown

            # Start Calibration
            self.perform_calibration()

            # Update button text
            self.captureButton1.setText("Capture Finished")
            self.captureButton1.setDisabled(True)

            # Emit the signal with the updated status text
            self.update_status_signal.emit("Capture Finished")

            # Update DONE button to Test Calibration
            self.doneButton1.setText("Test Calibration")
            self.doneButton1.clicked.connect(self.test_calibration)  # change connect to calibartion test

    def perform_calibration(self):
        # Camera calibration to return calibration parameters (camera_matrix, distortion_coefficients)
        print("Start Calibration")

        # Emit the signal with the updated status text
        self.update_status_signal.emit("Calibration in progess...")

        # Lists to store object points and image points from all the images.
        object_points = []  # 3D points in real world space
        image_points = []  # 2D points in image plane.

        # Load saved images and find chessboard corners
        image_files = sorted(os.listdir("output"))  # TODO replace output with variable
        for file_name in image_files:
            if file_name.startswith("corner_") and file_name.endswith(".png"):
                file_path = os.path.join("output", file_name)

                # Load the image
                frame = cv2.imread(file_path)

                # Detect corners
                ret_corners, corners, _ = self.detectCorners(frame, self.no_of_columns, self.no_of_rows)

                if ret_corners:
                    # Assuming that the chessboard is fixed on the (0, 0, 0) plane
                    object_points.append(
                        self.generate_object_points(self.no_of_columns, self.no_of_rows, self.square_size)
                    )
                    image_points.append(corners)

        # check with min_cap for minimum images needed for calibration (Default is 3)
        if len(object_points) >= self.min_cap:
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
                with open(
                    f"./output/intrinsic_{timestamp}.txt", "w"
                ) as file:  # TODO update output folder with variable
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
                with open(
                    f"./output/extrinsic_{timestamp}.txt", "w"
                ) as file:  # TODO update output folder with variable
                    for i in range(len(rvecs)):
                        file.write(f"\n\nImage {i+1}:\n")
                        file.write(f"Rotation Vector:\n{rvecs[i]}\n")
                        file.write(f"Translation Vector:\n{tvecs[i]}")

                self.outputWindow.setText(
                    f"Calibration parameters saved to ./output/intrinsic_{timestamp}.txt and ./output/extrinsic_{timestamp}.txt."
                )
                print(
                    f"Calibration parameters saved to ./output/intrinsic_{timestamp}.txt and ./output/extrinsic_{timestamp}.txt."
                )

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
        self.timer.start(100)  # Start camera feed

        if self.cal_imported == False:
            # Update DONE button to Test Calibration
            self.doneButton1.setText("Save to File")
            self.doneButton1.clicked.connect(self.save_calibration)  # change connect to calibartion test
        else:
            # Update DONE button to Test Calibration
            self.doneButton1.setText("Continue to De-Warp")
            self.doneButton1.clicked.connect(self.start_pwarp)  # change connect to calibartion test

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

        # print(f"Dumping below to file {fname}: \n\n {data}")

        try:
            # Write the calibration parameters to the JSON file
            with open(f"./output/{fname}", "w") as file:  # TODO update output folder with variable
                json.dump(data, file)

            # Emit the signal with the updated status text
            self.update_status_signal.emit("Calibration file Saved")

            # Update DONE button to Test Calibration
            self.doneButton1.setText("Continue to De-Warp")
            self.doneButton1.clicked.connect(self.start_pwarp)  # change connect to calibartion test

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
            self.image_dewarp = True  # TODO Does not below here

            # Load the image using OpenCv
            self.cv_image = cv2.imread(file_name)
            self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_RGB2BGR)

            # Covert to Pixmap
            pixmap = self.imageToPixmap(self.cv_image)  ## Test

            # Show stuff
            self.imageFrame.setPixmap(pixmap)
            self.imageFrame.setScaledContents(False)  # Set to false to prevent issues with aspect ratio

            # TODO check pixmap set is not none
            if self.imageFrame.pixmap() is not None:
                self.processoutputWindow.setText("Image loaded")

                # Prevent input until started
                self.imageFrame.mousePressEvent = None
                self.imageFrame.keyPressEvent = None

            else:
                self.processoutputWindow.setText("Problem displaying image")

    def start_pwarp(self):
        # Stop Camera
        self.timer.stop()  # Should not be here ! works for now since only images are used

        # Set Mouse Events

        # TODO Check if any pixmap is set before continuing
        # Check if the pixmap is set on the Image Frane
        if self.imageFrame.pixmap() is not None:
            # Temp disable Start button untill all 4 points are collected TODO
            self.startButtonPwarp.setDisabled(True)  # Should not be here !

            # Disable the first tab (Camera Calibration)
            self.tabs.setTabEnabled(0, False)  # Should not be here !

            # Switch to the second tab (Perspective-Warp)
            self.tabs.setCurrentIndex(1)  # Should not be here !

            if self.image_dewarp == True:
                print("Image Perspective Warp started")
                self.update_status_signal.emit("Image Perspective Warp started")

                frame = self.cv_image

                # Check if 'frame' is a valid NumPy array
                if isinstance(frame, np.ndarray):
                    # Disable Load image Button when import succeeded and dewarp started
                    self.loadImage.setDisabled(True)  # LoadImage / load_image is confusing TODO

                    if len(frame.shape) == 3:  # Check if it's a 3D array (color image)
                        self.warper_result = self.dewarp(frame)  # return warper

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

                        # TODO next steps - Tweak landmarks
                        self.startButtonPwarp.clicked.connect(self.tweak_pwarp)

                        if self.image_tuning_dewarp == False:
                            self.startButtonPwarp.clicked.disconnect(
                                self.start_pwarp
                            )  # disconnect the previous connection

                    else:
                        print("Invalid frame format: Not a 3D array (color image)")
                else:
                    print("Invalid frame format: Not a NumPy array")

            elif self.usb_dewarp == True:
                # usb_dewarp()
                print("Starting usb camera de-warp")  # -> update status

                # Start the camera again
                self.timer.start(100)  # Assuming 100 milliseconds per frame update (fast is 20)

                # TODO
                ret, frame = self.cap.read()  # read frame from webcam   -> use update_camera function

                if ret:  # if frame captured successfully
                    # frame_inverted = cv2.flip(frame, 1)  # flip frame horizontally
                    # original_inverted_frame = frame_inverted.copy()  # Store the original frame
                    # undistorted_frame = self.undistort_frame(frame, self.camera_matrix, self.camera_dist_coeff)
                    # dewarped_frame = self.warper.warp(undistorted_frame.copy())  # Perform dewarping
                    # self.display_dewarped_image(dewarped_frame)
                    pass  # temp

            elif self.network_dewarp == True:
                # network_dewarp()
                print("Starting network de-warp")  # -> update status
        else:
            print("No Image Loaded")
            self.processoutputWindow.setText("No Image loaded")
            QMessageBox.question(
                self, "Warning!", f"No Image detected.\n\nPlease make sure an Image is loaded", QMessageBox.Ok
            )

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
            # self.imageFrame.mousePressEvent = None

            # Start collecting arrow key events --> TEST TODO Ths is not the correct way
            self.imageFrame.keyPressEvent = self.keypress_tuning_event

        # Starts a loop collecting points
        while True:
            if len(self.points) == 0:
                self.processoutputWindow.setText(f"Select the first landmark {self.config.landmark1}")
                # Draw landmark 1 on 2d field view
                # print("Drawing landmark 1:", landmark1)
                self.field_image = self.draw_landmark(
                    self.field_image, self.config.field_coordinates_lm1, SoccerFieldColors.Red.value
                )

                # Convert to Pixman
                self.pixmap = self.imageToPixmap(self.field_image)
                pixmap = QPixmap(self.pixmap)

                # Load the image
                self.ProcessImage.setPixmap(pixmap)
                self.ProcessImage.setScaledContents(True)

            if len(self.points) == 1:
                self.processoutputWindow.setText(f"Select the second landmark {self.config.landmark2}")
                # Draw landmark 2 on 2d field view
                # print("Drawing landmark 2:", landmark2)
                self.field_image = self.draw_landmark(
                    self.field_image, self.config.field_coordinates_lm2, SoccerFieldColors.Red.value
                )

                # Convert to Pixman
                self.pixmap = self.imageToPixmap(self.field_image)
                pixmap = QPixmap(self.pixmap)

                # Load the image
                self.ProcessImage.setPixmap(pixmap)
                self.ProcessImage.setScaledContents(True)

            if len(self.points) == 2:
                self.processoutputWindow.setText(f"Select the third landmark {self.config.landmark3}")
                # Draw landmark 3 on 2d field view
                # print("Drawing landmark 3:", landmark3)
                self.field_image = self.draw_landmark(
                    self.field_image, self.config.field_coordinates_lm3, SoccerFieldColors.Red.value
                )

                # Convert to Pixman
                self.pixmap = self.imageToPixmap(self.field_image)
                pixmap = QPixmap(self.pixmap)

                # Load the image
                self.ProcessImage.setPixmap(pixmap)
                self.ProcessImage.setScaledContents(True)

            if len(self.points) == 3:
                self.processoutputWindow.setText(f"Select the last landmark {self.config.landmark4}")
                # Draw landmark 4 on 2d field view
                # print("Drawing landmark 4:", landmark4)
                self.field_image = self.draw_landmark(
                    self.field_image, self.config.field_coordinates_lm4, SoccerFieldColors.Red.value
                )

                # Convert to Pixman
                self.pixmap = self.imageToPixmap(self.field_image)
                pixmap = QPixmap(self.pixmap)

                # Load the image
                self.ProcessImage.setPixmap(pixmap)
                self.ProcessImage.setScaledContents(True)

            if len(self.points) == 4:
                # Enable Start button again when all 4 points are collected
                self.startButtonPwarp.setDisabled(False)

                # Stop mouse press event registration
                self.imageFrame.mousePressEvent = None

                if self.image_tuning_dewarp == True:
                    # tune self.point before processing them when tuning is enabled
                    # print("Start tuning landmarks")
                    # print(f"Tuning the selected Landmark: {self.points}")
                    # self.processoutputWindow.setText(f"Tuning selected Landmarks" )

                    # TODO - need a way to update field image on key event

                    self.startButtonPwarp.setText("Click when done tuning")
                    self.startButtonPwarp.clicked.connect(self.stop_tuning)

                break

            QApplication.processEvents()

        print(
            f"Check UI Frame | W: {self.width()}, H: {self.height()}, Supersample: {self.supersample}"
        )  # is using wrong values -> cameraFrame
        print(
            f"Check cameraFrame| W: {self.cameraFrame.width()}, H: {self.cameraFrame.height()}, Supersample: {self.supersample}"
        )  # is using wrong values -> cameraFrame

        # TODO is CameraFrame the best option?
        # TODO Why do we need to change the order of last two points?
        # Can we fix this sorting without hard-coding to the config?
        warper = Warper(
            points=np.array([self.points[0], self.points[1], self.points[3], self.points[2]]),
            landmark_points=self.landmark_points,
            width=self.cameraFrame.width(),
            height=self.cameraFrame.height(),
            supersample=self.supersample,
        )

        return warper

    def stop_tuning(self):
        self.image_tuning_dewarp == False

        # Disable input
        self.imageFrame.mousePressEvent = None
        self.imageFrame.keyPressEvent = None

        # Disable widget -> Maybe a bit much
        self.imageFrame.setEnabled(False)

        # Set save options TODO
        # For now Quit App
        self.startButtonPwarp.setText("DONE")
        self.startButtonPwarp.clicked.connect(
            self.close_application
        )  # close when done -> now closes directly without showing DONE

    def draw_landmark(self, image, landmark, color):
        # Draw the landmark
        cv2.circle(image, landmark, 15, (color), -1)  # Red dot for landmark
        cv2.putText(
            image,
            f"{landmark}",
            (landmark[0] + 20, landmark[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (SoccerFieldColors.LightGreen.value),
            2,
            cv2.LINE_AA,
        )  # light green coords

        return image

    def draw_landmark_selected(self, image, landmark, color):
        # Convert landmark to a tuple of integers
        landmark = (int(landmark[0]), int(landmark[1]))

        # Draw the landmark selector
        cv2.circle(image, landmark, 30, (color), 3)  # Meganta circle for landmark selection while tuning

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
                bgimage = cv2.rectangle(self.cv_image, (x, y), (x + 2, y + 2), (SoccerFieldColors.Green.value), 2)
                self.display_landmarks(bgimage)
            else:
                print("Pixmap is not set")

    def keypress_tuning_event(self, event):
        if len(self.points) == 4:  # Ensure 4 points are selected
            if event.key() == Qt.Key_Up:
                self.points[self.selected_point] = self.adjust_point(self.points[self.selected_point], "up")
            elif event.key() == Qt.Key_Down:
                self.points[self.selected_point] = self.adjust_point(self.points[self.selected_point], "down")
            elif event.key() == Qt.Key_Left:
                self.points[self.selected_point] = self.adjust_point(self.points[self.selected_point], "left")
            elif event.key() == Qt.Key_Right:
                self.points[self.selected_point] = self.adjust_point(self.points[self.selected_point], "right")
            elif event.key() == Qt.Key_1:
                self.selected_point = 0
                print("Selected landmark 1")
                self.processoutputWindow.setText(f"Landmark 1 selected")

                # print(f"Selected point index: {self.selected_point}")
                # print(f"self.landmark_points: {self.landmark_points}")
                # print(f"Value at selected point: {self.landmark_points[self.selected_point]}")

                # Draw selector on field image
                self.field_image_selected = self.draw_landmark_selected(
                    self.field_image.copy(), self.landmark_points[self.selected_point], MarkerColors.Yellow.value
                )

                # Convert to Pixman
                self.pixmap = self.imageToPixmap(self.field_image_selected)
                pixmap = QPixmap(self.pixmap)

                # Load the image
                self.ProcessImage.setPixmap(pixmap)
                self.ProcessImage.setScaledContents(True)

                # Cleanup
                self.field_image_selected = None

            elif event.key() == Qt.Key_2:
                self.selected_point = 1
                print("Selected landmark 2")
                self.processoutputWindow.setText(f"Landmark 2 selected")

                # print(f"Selected point index: {self.selected_point}")
                # print(f"self.landmark_points: {self.landmark_points}")
                # print(f"Value at selected point: {self.landmark_points[self.selected_point]}")

                # Draw selector on field image
                self.field_image_selected = self.draw_landmark_selected(
                    self.field_image.copy(), self.landmark_points[self.selected_point], MarkerColors.Yellow.value
                )

                # Convert to Pixman
                self.pixmap = self.imageToPixmap(self.field_image_selected)
                pixmap = QPixmap(self.pixmap)

                # Load the image
                self.ProcessImage.setPixmap(pixmap)
                self.ProcessImage.setScaledContents(True)

                # Cleanup
                self.field_image_selected = None

            elif event.key() == Qt.Key_3:
                self.selected_point = 2
                print("Selected landmark 3")
                self.processoutputWindow.setText(f"Landmark 3 selected")

                # print(f"Selected point index: {self.selected_point}")
                # print(f"self.landmark_points: {self.landmark_points}")
                # print(f"Value at selected point: {self.landmark_points[self.selected_point]}")

                # Draw selector on field image
                self.field_image_selected = self.draw_landmark_selected(
                    self.field_image.copy(), self.landmark_points[self.selected_point], MarkerColors.Yellow.value
                )

                # Convert to Pixman
                self.pixmap = self.imageToPixmap(self.field_image_selected)
                pixmap = QPixmap(self.pixmap)

                # Load the image
                self.ProcessImage.setPixmap(pixmap)
                self.ProcessImage.setScaledContents(True)

                # Cleanup
                self.field_image_selected = None

            elif event.key() == Qt.Key_4:
                self.selected_point = 3
                print("Selected landmark 4")
                self.processoutputWindow.setText(f"Landmark 4 selected")

                # print(f"Selected point index: {self.selected_point}")
                # print(f"self.landmark_points: {self.landmark_points}")
                # print(f"Value at selected point: {self.landmark_points[self.selected_point]}")

                # Draw selector on field image
                self.field_image_selected = self.draw_landmark_selected(
                    self.field_image.copy(), self.landmark_points[self.selected_point], MarkerColors.Yellow.value
                )

                # Convert to Pixman
                self.pixmap = self.imageToPixmap(self.field_image_selected)
                pixmap = QPixmap(self.pixmap)

                # Load the image
                self.ProcessImage.setPixmap(pixmap)
                self.ProcessImage.setScaledContents(True)

                # Cleanup
                self.field_image_selected = None

            else:
                super().keyPressEvent(event)  # Pass other key events to the base class

            # Make rectangle red when selected
            print(f"Updating perspective transform, using: self.points (pts{self.selected_point}): {self.points}")

            # Check if the pixmap is set on the Image Frane
            if self.imageFrame.pixmap() is not None:
                # TODO update selected point -> update perspective warp
                print("Update Image Frame / PWarp view")

                # Perform pwarp at every key press to update frame
                # self.warper_result = self.dewarp(self.cv_image.copy()) # return warper

                # Apply camera correction if any set
                undistorted_frame = self.undistort_frame(
                    self.cv_image.copy(), self.camera_matrix, self.camera_dist_coeff
                )

                # Perform pwarp at every key press to update frame
                self.warper_result = self.dewarp(undistorted_frame.copy())  # return warper

                # Perform dewarping
                dewarped_frame = self.warper_result.warp(undistorted_frame)

                # Update the display with the dewarped image
                self.display_dewarped_image(dewarped_frame)

            else:
                print("Pixmap is not set")

    def adjust_point(self, point, direction):
        """Adjust the point based on arrow key input"""
        x, y = point
        if direction == "up":
            print(f"Moved point {direction}")
            return (x, y - 1)
        elif direction == "down":
            print(f"Moved point {direction}")
            return (x, y + 1)
        elif direction == "left":
            print(f"Moved point {direction}")
            return (x - 1, y)
        elif direction == "right":
            print(f"Moved point {direction}")
            return (x + 1, y)
        return point

    # do i need this or can i use display_dewarped_image again
    # TODO this should just call the `Warper`
    def update_perspective_transform(self, pts1, pts2):
        self.frame = self.cv_image

        # Call the height and width method to get the actual value of the frame
        W = self.cameraFrame.width()  # is it cheating to use frame instead of img?
        H = self.cameraFrame.height()

        # Update and display the perspective-transformed image
        self.pts1 = np.array(pts1)
        self.landmark_points = np.array(pts2)

        M = cv2.getPerspectiveTransform(self.pts1.astype(np.float32), self.landmark_points.astype(np.float32))

        supersample = self.supersample

        if self.frame is None:
            self.frame = cv2.warpPerspective(self.frame, M, (W * supersample, H * supersample))
        else:
            self.frame[:] = cv2.warpPerspective(
                self.frame, M, (W * supersample, H * supersample)
            )  # -> issue using from frame.shape-

        # Convert the adjusted image to QPixmap and display it
        self.display_landmarks(self.frame)  # TODO

    def tweak_pwarp(self):
        # frame = self.cv_image
        # Print new instruction
        # self.processoutputWindow.setText("Tuning Landmarks started")

        # Add check if Tuning is started TODO
        self.image_tuning_dewarp = True  # Track if an image tuning is used for dewarping

        # Start de warp again for tuning
        self.start_pwarp()

    def close_application(self):
        QApplication.quit()
