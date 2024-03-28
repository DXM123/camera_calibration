import datetime
import json
import os

import cv2
import numpy as np
import glob # for load_images

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

    def __init__(self, parent: QMainWindow):
        super(QWidget, self).__init__(parent)

        self.config = get_config()
        self.min_cap = self.config.min_cap
        self.countdown_seconds = self.config.countdown_seconds # set to initial value

        # Index for loaded images
        self.image_files = []
        self.current_image_index = -1  # Start with -1, so the first image is at index 0

        # Object points for calibration
        self.objp = np.zeros((1, self.config.no_of_columns * self.config.no_of_rows, 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:self.config.no_of_columns, 0:self.config.no_of_rows].T.reshape(-1, 2)

        # Image  / frame characteristics 
        self.frame_shape = None
        self.object_points = []  # 3D points in real world space
        self.image_points = []   # 2D points in image plane

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
        self.cal_saved = False # Track if calibration is saved to file
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

        # Add dewarped frame
        self.dewarped_frame = None

        # Add field_image Temp
        self.field_image = None

        # Placeholder to count the ammount of images saved
        self.ImagesCounter = 0

        # Initialize camera_matrix as None
        self.camera_matrix = None

        # Initialize camera_distortion coefficients as None
        self.camera_dist_coeff = None

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

        self.cameraFrame.resize(640, 480) # Start vaue
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

        # Set fixed width for optionsFrame
        self.optionsFrame.setFixedWidth(500)

        # Add options widgets to optionsFrame:
        input_label = QLabel("Input Options:")
        input_label.setFixedHeight(48)
        self.optionsFrame.layout.addWidget(input_label)

        # Add radio buttons for selecting options
        self.input_layout = QHBoxLayout()

        self.input_camera = QRadioButton("CAM")
        self.input_network = QRadioButton("Network")
        self.input_images = QRadioButton("Images") 

        # Set "USB" as the default option
        self.input_camera.setChecked(True)

        # Connect the toggled signal of radio buttons to a slot function, not for images
        self.input_images.toggled.connect(self.update_capture_button_state)
        self.input_network.toggled.connect(self.update_capture_button_state)

        # Create a button group to make sure only one option is selected at a time
        self.input_group = QButtonGroup()
        self.input_group.addButton(self.input_camera)
        self.input_group.addButton(self.input_network)
        self.input_group.addButton(self.input_images)

        self.input_layout.addWidget(self.input_camera)
        self.input_layout.addWidget(self.input_network)
        self.input_layout.addWidget(self.input_images)

        self.optionsFrame.layout.addLayout(self.input_layout)

        #===============================================================
        # Add Camera Lens Options (Height and Fisheye toggle) (TODO)
        #===============================================================

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

        # Create Vertical Box Layout for tab2 inner frame
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

        # Store the initial size for later use
        self.initialTab1InnerSize = self.tab1inner.size()
        self.initialTab2InnerSize = self.tab2inner.size()
        print(f"tab2inner initial size: {self.initialTab2InnerSize}")
        print(f"tab1inner initial size: {self.initialTab1InnerSize}")

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

        # Set fixed width for processFrame
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


    def update_capture_button_state(self):
        # First, disconnect all previously connected signals to avoid multiple connections.
        try:
            self.captureButton1.clicked.disconnect()
        except TypeError:
            # If no connections exist, a TypeError is raised. Pass in this case.
            pass

        if self.input_images.isChecked():
            self.captureButton1.setEnabled(True)
            self.captureButton1.setText("Load Images")
            self.captureButton1.clicked.connect(self.select_directory_and_load_images)
        elif self.input_camera.isChecked():
            self.captureButton1.setEnabled(True)
            self.captureButton1.setText("Start Capture")
            self.captureButton1.clicked.connect(self.start_capture) 
        else:
            self.captureButton1.setEnabled(False)
            
    # Browse input folder for images 
    def select_directory_and_load_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", options=options)
        
        if directory:
            self.image_files = glob.glob(os.path.join(directory, '*.png')) + \
                               glob.glob(os.path.join(directory, '*.jpg')) + \
                               glob.glob(os.path.join(directory, '*.bmp'))
            self.image_files.sort()  # Optional: Sort the files

            # Print the files found
            print(f"Found {len(self.image_files)} files:")
            for file in self.image_files:
                print(file)

            # Disable Load Images button
            self.captureButton1.setEnabled(False)

            self.load_next_image()
    
    def load_next_image(self):
        self.current_image_index += 1
        if self.current_image_index < len(self.image_files):
            # Print which image is being loaded and its index
            print(f"Loading image {self.current_image_index + 1}/{len(self.image_files)}: {self.image_files[self.current_image_index]}")
            self.load_image_local(self.image_files[self.current_image_index])
        else:
            print("No more images in the directory.")

            # Disable input (TESTING)
            #self.cameraFrame.mousePressEvent = None
            #self.cameraFrame.keyPressEvent = None

            # GOTO NEXT -> Press DONE

    # Image loading for tab1
    def load_image_local(self, file_name):
        if file_name:
            # Image loading and processing logic here
            self.cv_image = cv2.imread(file_name)
            self.cv_image_rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)  # Corrected color conversion

            # Covert to Pixmap and other operations...
            pixmap = self.imageToPixmap(self.cv_image_rgb)

            # Set loaded image in CameraFrame
            self.cameraFrame.setPixmap(pixmap)
            self.cameraFrame.setScaledContents(False) # Never Scale

            # Adjust cameraFrame size to match the pixmap size - Set Fixed Size like imageFrame
            self.cameraFrame.setFixedSize(pixmap.size())

            # Allign Buttons to the same size self.startButtonPwarp & self.loadImage
            # Get the width of cameraFrame
            cameraFrameWidth = self.cameraFrame.size().width()

            # Print the cameraFrame width
            print(f"cameraFrame Width: {cameraFrameWidth}")

            # Set the buttons to the same width as cameraFrame
            self.doneButton1.setFixedWidth(cameraFrameWidth)
            self.captureButton1.setFixedWidth(cameraFrameWidth)

            if self.cameraFrame.pixmap() is not None:
                self.processoutputWindow.setText("Image loaded")
                
                self.update_image_feed(self.cv_image) 

            else:
                self.processoutputWindow.setText("Problem displaying image")

    # Image loading for tab2
    def load_image_forWarp(self, file_name):
        if file_name:
            # Image loading and processing logic here
            self.cv_image = cv2.imread(file_name)
            self.cv_image_rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)  # Corrected color conversion

            # Covert to Pixmap and other operations...
            pixmap = self.imageToPixmap(self.cv_image_rgb)

            # Set loaded image in CameraFrame
            self.imageFrame.setPixmap(pixmap)
            self.imageFrame.setScaledContents(False) # Never Scale

            # Adjust cameraFrame size to match the pixmap size - Set Fixed Size like imageFrame
            self.imageFrame.setFixedSize(pixmap.size())

            # Allign Buttons to the same size self.startButtonPwarp & self.loadImage
            # Get the width of cameraFrame
            imageFrameWidth = self.imageFrame.size().width()

            # Print the cameraFrame width
            print(f"cameraFrame Width: {imageFrameWidth}")

            # Set the buttons to the same width as cameraFrame
            self.doneButton1.setFixedWidth(imageFrameWidth)
            self.captureButton1.setFixedWidth(imageFrameWidth)

            if self.imageFrame.pixmap() is not None:
                self.processoutputWindow.setText("Image loaded")
                    
                self.update_image_feed(self.cv_image) 

            else:
                self.processoutputWindow.setText("Problem displaying image")

    # This is working when using general name 
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Space):
            self.load_next_image()

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
    
    def update_image_feed(self, image):
        # Save original image
        org_image = image.copy()

        #print(f"test_calibration is set to: {self.test_started}")

        if self.test_started or self.cal_imported:
            # Print loaded camera matix
            #self.outputWindow.setText(f"Camera matrix used for testing:{self.camera_matrix}")

            #TODO Check self.camera.matrix and self.dist_coeff are available?
            print("Calibartion test started or calibration imported")

            # Debug print for camera matrix and distortion coefficients
            #print(f"Camera Matrix used for Testing:\n{self.camera_matrix}")
            #print(f"Distortion Coefficients used for Testing:\n{self.camera_dist_coeff}")

            # prep for fisheye toggle
            if self.input_images.isChecked():
                undistorted_frame = self.undistort_fisheye_frame(image, self.camera_matrix, self.camera_dist_coeff)
            else:
                undistorted_frame = self.undistort_frame(image, self.camera_matrix, self.camera_dist_coeff)

            undistorted_frame_rgb = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
            self.pixmap = self.imageToPixmap(undistorted_frame_rgb)
            
            # TODO Use cameraFrame for tab1 and imageFrame for tab2
            #if self.tab1.isActiveWindow():
            #    self.cameraFrame.setPixmap(self.pixmap)
            #else:
            #    self.imageFrame.setPixmap(self.pixmap)

            if self.tabs.currentIndex() == 0:  # tab1 is at index 0
                self.cameraFrame.setPixmap(self.pixmap)
            elif self.tabs.currentIndex() == 1:  # tab2 is at index 1
                self.imageFrame.setPixmap(self.pixmap)

        else:
            print("No test started or calibration file loaded")
            #org_image = self.imageToPixmap(image)
            ret_corners, corners, frame_with_corners = self.detectCorners(image, self.no_of_columns, self.no_of_rows)

            if ret_corners:
                # Display the image with corners
                frame_with_corners = cv2.cvtColor(frame_with_corners, cv2.COLOR_BGR2RGB)
                self.pixmap = self.imageToPixmap(frame_with_corners)
                self.cameraFrame.setPixmap(self.pixmap)
                # Only save when not testing
                if self.test_started != True:
                    self.save_screenshot(org_image)  # Save original Frame
            else:
                # Display the original image
                org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB) 
                self.pixmap = self.imageToPixmap(org_image)
                self.cameraFrame.setPixmap(self.pixmap)

        # Ensure the image does not scales with the label -> issue with aspect ratio TODO
        self.cameraFrame.setScaledContents(False)
        self.update()

    def update_camera_feed(self):
        # This method will be called at regular intervals by the timer
        ret, frame = self.cap.read()  # read frame from webcam

        #print(f"test_calibration is set to: {self.test_started}")

        if ret:  # if frame captured successfully
            frame_inverted = cv2.flip(frame, 1)  # flip frame horizontally --> Is this needed TODO
            original_inverted_frame = frame_inverted.copy()  # Store the original frame

            # TODO update inverted_frame to corrected frame
            if self.test_started or self.cal_imported:

                # Print loaded camera matix
                #self.outputWindow.setText(f"Camera matrix used for testing:{self.camera_matrix}")

                #TODO Check self.camera.matrix and self.dist_coeff are available?

                # Debug print for camera matrix and distortion coefficients
                #print(f"Camera Matrix used for Testing:\n{self.camera_matrix}")
                #print(f"Distortion Coefficients used for Testing:\n{self.camera_dist_coeff}")

                if self.input_images.isChecked():
                    undistorted_frame = self.undistort_fisheye_frame(frame, self.camera_matrix, self.camera_dist_coeff)
                else:
                    undistorted_frame = self.undistort_frame(frame, self.camera_matrix, self.camera_dist_coeff)
                    
                frame_inverted = undistorted_frame # cheesey replace

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
                    self.countdown_seconds = self.config.countdown_seconds  # Reset the countdown after saving

                if ret_corners:
                    # Display the frame with corners
                    self.pixmap = self.CameraToPixmap(frame_with_corners)
                    self.cameraFrame.setPixmap(self.pixmap)
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
            self.countdown_seconds = self.config.countdown_seconds
            self.countdown_timer.start()

    def detectCorners(self, image, columns, rows):
        # Stop the iteration when specified accuracy, epsilon, is reached or specified number of iterations are completed. 
        # In this case the maximum number of iterations is set to 30 and epsilon = 0.1
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001) # We want a high accuracy, but does it really help?

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
            # Now Store Object Points 
            self.object_points.append(self.objp)

            # Refining pixel coordinates for given 2d points. A larger search window means the algorithm considers a broader area around each corner for refining its location. 
            # cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
            #cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

            ## Now Store Corners Detected
            self.image_points.append(corners)

            # draw and display the chessboard corners
            cv2.drawChessboardCorners(image, (columns, rows), corners, ret)
        
            # Print the number of corners found
            print("Number of corners found:", len(corners))

            # Display the x and y coordinates of each corner _> move to perform_calibration to also get filename
            #for i, corner in enumerate(corners):
            #    x, y = corner.ravel() # Converts the corner's array to a flat array and then unpacks the x and y values
            #    print(f"Found Corner {i+1}: x = {x}, y = {y}")

        return ret, corners, image

    def save_screenshot(self, frame):
        # Ensure that the output directory exists
        tmp_dir = self.config.tmp_data

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        # Generate a timestamp for the screenshot filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

        filename = f"{tmp_dir}/corner_{timestamp}.png"

        # Save the frame as an image
        cv2.imwrite(filename, frame)

        self.ImagesCounter += 1

        # Update the output window with the saved filename
        self.outputWindow.setText(f"Screenshot saved:\n{filename}\nTotal Images Collected: {self.ImagesCounter}")
        #print(f"Frame Shape = {frame.shape[:2]}") # wrong way around
        print(f"Frame Shape = (Width: {frame.shape[1]}, Height: {frame.shape[0]})")

        print(f"Screenshot saved:\n {filename}\nTotal Images Collected: {self.ImagesCounter}")

    # Check collected images with corners and start calibration if ok
    def check_done(self):
        if self.ImagesCounter < self.config.min_cap:
            rem = self.config.min_cap - self.ImagesCounter
            QMessageBox.question(
                self,
                "Warning!",
                f"The minimum number of captured images is set to {self.config.min_cap}.\n\nPlease collect {rem} more images",
                QMessageBox.Ok,
            )
        else:
            self.timer.stop()  # Stop camera feed
            self.countdown_timer.stop()  # Stop countdown

            # Update button text
            self.captureButton1.setText("Capture Finished")
            self.captureButton1.setDisabled(True)

            # Emit the signal with the updated status text
            self.update_status_signal.emit("Capture Finished")

            # First, disconnect all previously connected signals to avoid multiple connections.
            try:
                self.doneButton1.clicked.disconnect()
            except TypeError:
                # If no connections exist, a TypeError is raised. Pass in this case.
                pass

            # Update DONE button to Test Calibration
            self.doneButton1.setText("Test Calibration")
            self.doneButton1.clicked.connect(self.test_calibration)  # change connect to calibartion test

    def perform_calibration(self):
        print(f"Test_calibration is set to: {self.test_started}")
        print("Start Calibration")
        self.update_status_signal.emit("Calibration in progress...")

        # At this stage object_points and image_points are alread available when detect_corners was ran for loading images / frames etc
        # Clear self.object_points and self.image_points to detect again?
        self.object_points = []  # 3D points in real world space
        self.image_points = []   # 2D points in image plane

        # Now using tmp data folder instead of input folder
        image_files = sorted(os.listdir(self.config.tmp_data))

        for file_name in image_files:
            if file_name.startswith("corner_") and file_name.endswith(".png"):
                file_path = os.path.join(self.config.tmp_data, file_name)
                frame = cv2.imread(file_path)

                # Detect Corners using OpenCV
                ret_corners, corners, _ = self.detectCorners(frame, self.no_of_columns, self.no_of_rows)

                if ret_corners:
                    #self.object_points.append(self.generate_object_points(no_of_columns, no_of_rows, square_size))
                    #self.image_points.append(corners) # -> done in detectCorners

                    # Display the x and y coordinates of each corner
                    for i, corner in enumerate(corners):
                        x, y = corner.ravel() # Converts the corner's array to a flat array and then unpacks the x and y values
                        print(f"Found Corner {i+1}: x = {x}, y = {y} in {file_path}")

                    # TODO Collect the Corners to be saved to corners.vnl
                    
                    # TODO Set boolean to calibration finished -> to prefent double / tripple run when moving to dewarp tab2

                    print("Calibration Succesfull")

        # check with min_cap for minimum images needed for calibration (Default should be 10) (3 for now)
        if len(self.object_points) >= self.config.min_cap:
            print(f"Found {self.object_points} images with corners detected")
            # Generate a timestamp for the screenshot filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            # Perform camera calibration
            ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(
                self.object_points, self.image_points, (frame.shape[1], frame.shape[0]), None, None
            )

            if ret:  # if calibration was successfully
                # Display the calibration results
                self.outputWindow.setText(f"Camera matrix:{camera_matrix}")
                print(f"Camera matrix found:{camera_matrix}")

                # Assign camera_matrix to the instance variable
                self.camera_matrix = camera_matrix

                self.outputWindow.setText(f"Distortion coefficient:{distortion_coefficients}")
                print(f"Distortion coefficient found:{distortion_coefficients}")

                # Assign camera_distortion coefficient to the instance variable
                self.camera_dist_coeff = distortion_coefficients

                # Save intrinsic parameters to intrinsic.txt
                if self.test_started == True:
                    with open(f"./{self.config.tmp_data}/intrinsic_{timestamp}.txt", "w") as file:
                        file.write("Camera Matrix:\n")
                        file.write(str(self.camera_matrix))
                        file.write("\n\nDistortion Coefficients:\n")
                        file.write(str(self.camera_dist_coeff))

                    self.outputWindow.setText(f"Rotation Vectors:{rvecs}")
                    print("\n Rotation Vectors:")
                    print(rvecs)

                    self.outputWindow.setText(f"Translation Vectors:{tvecs}")
                    print("\n Translation Vectors:")
                    print(tvecs)

                    # Save extrinsic parameters to extrinsic.txt
                    with open(f"./{self.config.tmp_data}/extrinsic_{timestamp}.txt", "w") as file:
                        for i in range(len(rvecs)):
                            file.write(f"\n\nImage {i+1}:\n")
                            file.write(f"Rotation Vector:\n{rvecs[i]}\n")
                            file.write(f"Translation Vector:\n{tvecs[i]}")
                    
                    self.outputWindow.setText(f"Calibration parameters saved to {self.config.tmp_data}/intrinsic_{timestamp}.txt and {self.config.tmp_data}/extrinsic_{timestamp}.txt.")
                    print(f"Calibration parameters saved to {self.config.tmp_data}/intrinsic_{timestamp}.txt and {self.config.tmp_data}/extrinsic_{timestamp}.txt.")

            else:
                print("Camera Calibration failed")
        else:
            print(f"Need {self.config.min_cap} images with corners, only {len(self.object_points)} found")
        

    
    #######################################################################

    def perform_calibration_fisheye(self):
        print(f"test_calibration is set to: {self.test_started}")
        print("Start Calibration")
        self.update_status_signal.emit("Calibration in progress...")

        # At this stage object_points and image_points are alread available when detect_corners was ran for loading images / frames etc
        # Clear self.object_points and self.image_points to detect again?
        self.object_points = []  # 3D points in real world space
        self.image_points = []   # 2D points in image plane

        # Now using tmp folder instead of input folder
        image_files = sorted(os.listdir(self.config.tmp_data))
        
        for file_name in image_files:
            if file_name.startswith("corner_") and file_name.endswith(".png"):
                file_path = os.path.join(self.config.tmp_data, file_name)
                frame = cv2.imread(file_path)
                ret_corners, corners, _ = self.detectCorners(frame, self.config.no_of_columns, self.config.no_of_rows)

                if ret_corners:
                    #self.object_points.append(self.generate_object_points(no_of_columns, no_of_rows, square_size))
                    #self.image_points.append(corners) --> Done in detectCorners

                    # Display the x and y coordinates of each corner
                    for i, corner in enumerate(corners):
                        x, y = corner.ravel() # Converts the corner's array to a flat array and then unpacks the x and y values
                        print(f"Found Corner {i+1}: x = {x}, y = {y} in {file_path}")

                    # TODO Collect the Corners to be saved to corners.vnl
                        
                    # TODO Set boolean to calibration finished -> to prefent double / tripple run when moving to dewarp tab2

                    print("Calibration Succesfull")

        if len(self.object_points) >= self.config.min_cap:  # Ensure there's enough data for calibration
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            obj_points = np.array(self.object_points)
            img_points = np.array(self.image_points)
            _img_shape = frame.shape[:2]
            
            calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

            N_OK = len(obj_points)
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
            tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

            try:
                rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                    obj_points,
                    img_points,
                    _img_shape[::-1],
                    K,
                    D,
                    rvecs,
                    tvecs,
                    calibration_flags,
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                )

                self.outputWindow.setText(f"Camera matrix found:{K}\nDistortion coefficients found:{D}")
                print(f"Camera matrix found:{K}\nDistortion coefficients found:{D}")
                print(f"The RMS Error is: {rms}")

                self.camera_matrix = K
                self.camera_dist_coeff = D

                ########################COPY FROM ABOVE################################

                # Save intrinsic parameters to intrinsic.txt
                #if self.test_started == False:
                if self.test_started == True: # TODO Not triggered when set to False
                    with open(f"{self.config.tmp_data}/intrinsic_{timestamp}.txt", "w") as file:
                        file.write("Camera Matrix:\n")
                        file.write(str(self.camera_matrix))
                        file.write("\n\nDistortion Coefficients:\n")
                        file.write(str(self.camera_dist_coeff))

                    self.outputWindow.setText(f"Rotation Vectors:{rvecs}")
                    print("\n Rotation Vectors:")
                    print(rvecs)

                    self.outputWindow.setText(f"Translation Vectors:{tvecs}")
                    print("\n Translation Vectors:")
                    print(tvecs)

                    # Save extrinsic parameters to extrinsic.txt
                    with open(f"{self.config.tmp_data}/extrinsic_{timestamp}.txt", "w") as file:
                        for i in range(len(rvecs)):
                            file.write(f"\n\nImage {i+1}:\n")
                            file.write(f"Rotation Vector:\n{rvecs[i]}\n")
                            file.write(f"Translation Vector:\n{tvecs[i]}")
                    
                    self.outputWindow.setText(f"Calibration parameters saved to {self.config.tmp_data}/intrinsic_{timestamp}.txt and {self.config.tmp_data}/extrinsic_{timestamp}.txt.")
                    print(f"Calibration parameters saved to {self.config.tmp_data}/intrinsic_{timestamp}.txt and {self.config.tmp_data}/extrinsic_{timestamp}.txt.")

                    ###########################

            except cv2.error as e:
                print(f"Calibration failed with error: {e}")

        else:
            print(f"Need at least {self.config.min_cap} images with corners for calibration. Only {len(self.object_points)} found")

    ######################################################################


    def generate_object_points(self, columns, rows, square_size):
        objp = np.zeros((1, columns * rows, 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size
        return objp

    def test_calibration(self):
        # TODO if Pause button was pressed , camera stops !!!!
        print("Testing Calibration")

        # Set test boolean to prevent mutiple saves (or 10 rows lower?)
        self.test_started = True

        # Start Calibration again also allow it to save intrinsic / extrinsic tmp files once -> Not working due to test_started value = True
        if self.input_images.isChecked():
            self.perform_calibration_fisheye()
            self.current_image_index = -1
            self.load_next_image() 

        if self.input_camera.isChecked():
            self.timer.start(100) # Start camera feed
            self.perform_calibration()

        # Set test boolean to prevent mutiple saves
        #self.test_started = True

        # Emit the signal with the updated status text
        self.update_status_signal.emit("Testing in progess....")

        # First, disconnect all previously connected signals to avoid multiple connections.
        try:
            self.doneButton1.clicked.disconnect()
        except TypeError:
            # If no connections exist, a TypeError is raised. Pass in this case.
            pass

        if self.cal_imported == False and self.cal_saved == False: # TRY to prevent double save
            # Update Test Calibration to Save to File
            self.doneButton1.setText("Save to File")
            self.doneButton1.clicked.connect(self.save_calibration)
        else:
            # Update DONE button to Test Calibration
            self.doneButton1.setText("Continue to De-Warp")
            self.doneButton1.clicked.connect(self.start_pwarp)

    def undistort_frame(self, frame, camera_matrix, distortion_coefficients):
        # Check if camera_matrix is available
        if self.camera_matrix is not None and self.camera_dist_coeff is not None:
            # Undistort the frame using the camera matrix and distortion coefficients
            undistorted_frame = cv2.undistort(frame, camera_matrix, distortion_coefficients)

            return undistorted_frame

        else:
            print("No camera matrix or distortion coefficient detected, showing original frame")

            return frame

    ############################################################################################
        
    # When lens field of view is above 160 degree we need fisheye undistort function for opencv
    def undistort_fisheye_frame_prev(self, frame, camera_matrix, distortion_coefficients):
        # Check if camera_matrix and distortion_coefficients are available
        if camera_matrix is not None and distortion_coefficients is not None:

            # Store Frame Dimension
            DIM = frame.shape[:2][::-1]  # Original image dimensions
            #print(f"Frame Dimension (DIM) = {DIM}")

            # Undistort and remap frame
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, distortion_coefficients, np.eye(3), camera_matrix, DIM, cv2.CV_16SC2)
            undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            return undistorted_frame
        else:
            print("No camera matrix or distortion coefficient detected, showing original frame")
            return frame
        
    def undistort_fisheye_frame_simple(self, frame, camera_matrix, distortion_coefficients, balance=0.0):
        """
        Undistort a frame captured with a fisheye lens.
        
        :param frame: Input frame with distortion.
        :param camera_matrix: Camera matrix (K).
        :param distortion_coefficients: Distortion coefficients (D).
        :param balance: Balance between the viewing area and undistortion effects, where 0 means full correction with possible cropping and 1 means minimal correction with all pixels retained in the output.
        :return: Undistorted frame.
        """
        if camera_matrix is None or distortion_coefficients is None:
            print("No camera matrix or distortion coefficient detected, showing original frame")
            return frame

        dim = frame.shape[:2][::-1]  # Original image dimensions
        #print(f"Frame Dimension input (DIM1) = {dim}")

        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix, distortion_coefficients, dim, np.eye(3), balance=balance
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, distortion_coefficients, np.eye(3), new_camera_matrix, dim, cv2.CV_16SC2
        )

        # Issue: The borderMode=cv2.BORDER_CONSTANT parameter in cv2.remap fills the areas outside the remapping with a constant value (default is black), which, 
        # if most of the image is out of bounds, results in a black image.

        undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        return undistorted_frame
    
        
    def undistort_fisheye_frame(self, frame, camera_matrix, distortion_coefficients, balance=0.0, dim2=None, dim3=None):
        """
        Undistort a frame captured with a fisheye lens.
        
        :param frame: Input frame with distortion.
        :param camera_matrix: Camera matrix (K).
        :param distortion_coefficients: Distortion coefficients (D).
        :param balance: Balance between the viewing area and undistortion effects, where 0 means full correction with possible cropping and 1 means minimal correction with all pixels retained in the output.
        :param dim2: Desired image size after undistortion (width, height). Default is the size of the input image.
        :param dim3: Size of the input image (width, height).
        :return: Undistorted frame.
        """
        if camera_matrix is None or distortion_coefficients is None:
            print("No camera matrix or distortion coefficient detected, showing original frame")
            return frame
        
        # DIM and dim1 ae always the same so why bother with scaling??
        _img_shape = frame.shape[:2]
        DIM=_img_shape[::-1] # should be original image shape used for calibration
        #print(f"Frame Dimension (DIM) = {DIM}")

        dim1 = frame.shape[:2][::-1]  # Original image dimensions of target images , but not the case here
        #print(f"Frame Dimension input (DIM1) = {dim1}")

        if not dim2:
            dim2 = dim1 # Target dimension for the new camera matrix optimization
        if not dim3:
            dim3 = dim1 # Output image dimension for undistortion map

        # Scale fx and cx with the width ratio, and fy and cy with the height ratio
        width_ratio = dim1[0] / DIM[0]
        height_ratio = dim1[1] / DIM[1]

        scaled_K = camera_matrix.copy()
        scaled_K[0, 0] *= width_ratio  # Scale fx
        scaled_K[1, 1] *= height_ratio  # Scale fy
        scaled_K[0, 2] *= width_ratio  # Scale cx
        scaled_K[1, 2] *= height_ratio  # Scale cy
        scaled_K[2, 2] = 1.0  # Keep the scale factor for homogeneous coordinates as 1

        # Now, estimate the new camera matrix optimized for dim2 dimensions
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, distortion_coefficients, dim2, np.eye(3), balance=balance)

        # Initialize the undistort rectify map for the dimensions of the output image (dim3)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, distortion_coefficients, np.eye(3), new_K, dim3, cv2.CV_16SC2)

        # Finally, remap the image using the undistortion map for the corrected image
        undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        return undistorted_frame
        
    #################################################################################################


    def save_calibration(self):
        # TODO verify why save_calibration is called when CTRL-D is pressed or when going to De-warp next step 
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

        try:
            print(f"Saving to file {fname}: \n\n {data}")

            # Write the calibration parameters to the JSON file
            with open(f"{self.config.tmp_data}/{fname}", "w") as file:
                json.dump(data, file)

            # Emit the signal with the updated status text
            self.update_status_signal.emit(f"Calibration file {fname} Saved")
            print(f"Calibartion file {fname} saved")

            # Self calibration saved to True
            self.cal_saved = True

            # First, disconnect all previously connected signals to avoid multiple connections.
            try:
                self.doneButton1.clicked.disconnect()
            except TypeError:
                # If no connections exist, a TypeError is raised. Pass in this case.
                pass

            # Update DONE button to Test Calibration
            self.doneButton1.setText("Continue to De-Warp")
            self.doneButton1.clicked.connect(self.start_pwarp)  # change connect to calibartion test

            # Stop Camera feed
            self.timer.stop()

            # Stop Input
            self.cameraFrame.keyPressEvent = None
            self.cameraFrame.mousePressEvent = None
            self.cameraFrame.releaseMouse()

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

            print("Image Loaded ....")
            print(f"Calibration imported is: {self.cal_imported}")

            #####################################################################################
            # Add up update_image_feed
            # TODO: Space still working for looping through images only 1 image should be used?
            # Clear the variable containing the list of images?

            self.load_image_forWarp(file_name)

            #####################################################################################

            # Load the image using OpenCv
            #self.cv_image = cv2.imread(file_name)
            #self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_RGB2BGR)

            # Covert to Pixmap
            #pixmap = self.imageToPixmap(self.cv_image)  ## Test

            # Show stuff
            #self.imageFrame.setPixmap(pixmap)
            #self.imageFrame.setScaledContents(False)  # Set to false to prevent issues with aspect ratio

            # Adjust imageFrame size to exactly match the pixmap size needed for proper mouse x,y
            #self.imageFrame.setFixedSize(pixmap.size())

            # Allign Buttons to the same size self.startButtonPwarp & self.loadImage
            # Get the width of cameraFrame
            #imageFrameWidth = self.imageFrame.size().width()

            # Print the cameraFrame width
            #print(f"imageFrame Width: {imageFrameWidth}")

            # Set the buttons to the same width as cameraFrame
            #self.startButtonPwarp.setFixedWidth(imageFrameWidth)
            #self.loadImage.setFixedWidth(imageFrameWidth)

            # TODO check pixmap set is not none
            if self.imageFrame.pixmap() is not None:
                self.processoutputWindow.setText("Image loaded")

                # Prevent input until started
                #self.imageFrame.mousePressEvent = None
                #self.imageFrame.keyPressEvent = None

            else:
                self.processoutputWindow.setText("Problem displaying image")

    def start_pwarp(self):
        # Stop Camera -> also stopped after save
        #self.timer.stop() # Should not be here ! works for now since only images are used

        # Check if the pixmap is set on the Image Frane
        if self.imageFrame.pixmap() is not None:

            # Temp disable Start button untill all 4 points are collected TODO
            self.startButtonPwarp.setDisabled(True)  # Should not be here !

            if self.image_dewarp == True:
                print("Image Perspective Warp started")
                self.update_status_signal.emit("Image Perspective Warp started")

                frame = self.cv_image

                # Check if 'frame' is a valid NumPy array
                if isinstance(frame, np.ndarray):
                    # Disable Load image Button when import succeeded and dewarp started
                    self.loadImage.setDisabled(True)  # LoadImage / load_image is confusing TODO

                    if len(frame.shape) == 3:  # Verify if valid frame shape

                        self.warper_result = self.dewarp(frame)  # return warper

                        if self.input_images.isChecked():
                            undistorted_frame = self.undistort_fisheye_frame(frame, self.camera_matrix, self.camera_dist_coeff)
                        else:
                            undistorted_frame = self.undistort_frame(frame, self.camera_matrix, self.camera_dist_coeff)

                        # Perform dewarping
                        self.dewarped_frame = self.warper_result.warp(undistorted_frame.copy())

                        # Update the display with the dewarped image
                        self.display_dewarped_image(self.dewarped_frame)

                        # Print stuff and update status bar
                        print("Dewarping process completed.")
                        self.update_status_signal.emit("Dewarping process completed.")

                        ## Check if tweak already started to prevent overwrite button text
                        if not self.image_tuning_dewarp == True:
                            # Update button text for next step
                            self.startButtonPwarp.setText("Tweak Landmarks")

                            # Tweak landmarks
                            self.startButtonPwarp.clicked.connect(self.tweak_pwarp)

                            # disconnect the previous connection
                            self.startButtonPwarp.clicked.disconnect(self.start_pwarp)

                        # TODO next steps - Tweak landmarks
                        #self.startButtonPwarp.clicked.connect(self.tweak_pwarp)

                        #if self.image_tuning_dewarp == False:
                        #    self.startButtonPwarp.clicked.disconnect(
                        #        self.start_pwarp
                        #    )  # disconnect the previous connection

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

                #if ret:  # if frame captured successfully
                    # frame_inverted = cv2.flip(frame, 1)  # flip frame horizontally
                    # original_inverted_frame = frame_inverted.copy()  # Store the original frame
                    # undistorted_frame = self.undistort_frame(frame, self.camera_matrix, self.camera_dist_coeff)
                    # dewarped_frame = self.warper.warp(undistorted_frame.copy())  # Perform dewarping
                    # self.display_dewarped_image(dewarped_frame)
                #    pass  # temp

            elif self.network_dewarp == True:
                # network_dewarp()
                print("Starting network de-warp")  # -> update status
        else:
            # Disable the first tab (Camera Calibration)
            self.tabs.setTabEnabled(0, False) # Should not be here !

            #########################################################################

            # Clear Image from self.imageFrame
            self.cameraFrame.setPixmap(QPixmap())
            self.imageFrame.setPixmap(QPixmap()) 

            # Switch to the second tab (Perspective-Warp)
            self.tabs.setCurrentIndex(1) # Should not be here !

            self.initialTab1InnerSizeCheck = self.tab1inner.size()
            self.initialTab2InnerSizeCheck = self.tab2inner.size()
            print(f"tab2inner current size: {self.initialTab2InnerSizeCheck}")
            print(f"tab1inner current size: {self.initialTab1InnerSizeCheck}")

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
                #self.imageFrame.mousePressEvent = None
                if not self.image_tuning_dewarp == True:
                    
                    # Stop mouse press event registration
                    self.imageFrame.mousePressEvent = None

                if self.image_tuning_dewarp == True:
                    # tune self.point before processing them when tuning is enabled
                    # print("Start tuning landmarks")
                    # print(f"Tuning the selected Landmark: {self.points}")
                    # self.processoutputWindow.setText(f"Tuning selected Landmarks" )

                    # TODO - need a way to update field image on key event

                    self.startButtonPwarp.setText("Click when done tuning")   ## Not updating TODO
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
        #self.imageFrame.mousePressEvent = None
        #self.imageFrame.keyPressEvent = None

        # Disable widget -> Maybe a bit much
        self.imageFrame.setEnabled(False)

        # Set save options TODO
        self.startButtonPwarp.setText("Save to binary file")
        self.startButtonPwarp.clicked.connect(self.save_prep_mat_binary)

        # For now Quit App
        #self.startButtonPwarp.setText("DONE")
        #self.startButtonPwarp.clicked.connect(
        #    self.close_application
        #)  # close when done -> now closes directly without showing DONE

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
            # print(f"Mouse clicked at x: {x}, y: {y}")

            # Debug Mouse events
            print(f"Clicked at local coordinates: ({event.x()}, {event.y()})")
            globalPos = self.mapToGlobal(event.pos())
            print(f"Clicked at global coordinates: ({globalPos.x()}, {globalPos.y()})")
            super().mousePressEvent(event)  # Call the parent class's event handler

            # Check if the pixmap is set on the Image Frane
            if self.imageFrame.pixmap() is not None:
                self.points.append((x, y))
                self.imageFrame.setPixmap(QPixmap())  # Clear the current pixmap -> Might cause flickering
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
                #self.selected_point = 3 # should be 2 TODO
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
                #self.selected_point = 2 # should be 3 TODO
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

                # Debug print for camera matrix and distortion coefficients
                print(f"Camera Matrix:\n{self.camera_matrix}")
                print(f"Distortion Coefficients:\n{self.camera_dist_coeff}")

                # Apply camera correction if any set ( Add Selection opion for Fish-Eye TODO)
                #undistorted_frame = self.undistort_frame(self.cv_image.copy(), self.camera_matrix, self.camera_dist_coeff)
                #undistorted_frame = self.undistort_fisheye_frame(self.cv_image.copy(), self.camera_matrix, self.camera_dist_coeff)

                # Not really needed since images now always use fisheye, but prep for fisheye toggle
                if self.input_images.isChecked():
                    undistorted_frame = self.undistort_fisheye_frame(self.cv_image.copy(), self.camera_matrix, self.camera_dist_coeff)
                else:
                    undistorted_frame = self.undistort_frame(self.cv_image.copy(), self.camera_matrix, self.camera_dist_coeff)

                # Perform pwarp at every key press to update frame
                self.warper_result = self.dewarp(undistorted_frame.copy())  # return warper

                # Perform dewarping
                self.dewarped_frame = self.warper_result.warp(undistorted_frame)

                # Update the display with the dewarped image
                self.display_dewarped_image(self.dewarped_frame)

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

    ####################### Binary file stuff WIP ###################
        
    def write_mat_binary_old(self, ofs, out_mat):
        if not ofs:
            return False
        if out_mat is None:
            s = 0
            ofs.write(s.to_bytes(4, byteorder='little'))
            return True
        rows, cols = out_mat.shape[:2]
        dtype = out_mat.dtype
        ofs.write(rows.to_bytes(4, byteorder='little'))
        ofs.write(cols.to_bytes(4, byteorder='little'))
        ofs.write(np.uint32(dtype).tobytes())
        ofs.write(out_mat.tobytes())
        return True
    
    def write_mat_binary(self, ofs, out_mat):
        if not ofs:
            return False
        if out_mat is None:
            s = 0
            ofs.write(s.to_bytes(4, byteorder='little'))
            return True
        rows, cols = out_mat.shape[:2]
        dtype = out_mat.dtype.itemsize  # Corrected line
        ofs.write(rows.to_bytes(4, byteorder='little'))
        ofs.write(cols.to_bytes(4, byteorder='little'))
        ofs.write(np.uint32(dtype).tobytes())
        ofs.write(out_mat.tobytes())
        return True
    
    ##################################################################
    
    def save_prep_mat_binary(self):
        # Example usage:
        # mat = cv2.imread('image.jpg')
        filename = "mat.bin"
        mat = self.dewarped_frame
        # self.save_mat_binary('mat.bin', mat)
        # loaded_mat = cv2.imread('mat.bin')
        #print ("Saving to binary file <name> and saving Mat image")
        #return self.save_mat_binary('mat.bin', mat)
        print(f"Saving to binary file {filename} and saving Mat image")
        return self.save_mat_binary(filename, mat)

    #################################################################

    def save_mat_binary(self, filename, output):
        with open(filename, 'wb') as ofs:
            return self.write_mat_binary(ofs, output)


    def read_mat_binary(self, ifs, in_mat):
        if not ifs:
            return False
        rows = int.from_bytes(ifs.read(4), byteorder='little')
        if rows == 0:
            return True
        cols = int.from_bytes(ifs.read(4), byteorder='little')
        dtype = np.dtype(int(np.uint32.frombytes(ifs.read(4), byteorder='little')))
        in_mat.release()
        in_mat.create(rows, cols, dtype)
        in_mat.data = ifs.read(in_mat.elemSize() * in_mat.total())
        return True

    def load_mat_binary(self, filename, output):
        with open(filename, 'rb') as ifs:
            return self.read_mat_binary(ifs, output)

    # Example usage:
    # mat = cv2.imread('image.jpg')
    # self.save_mat_binary('mat.bin', mat)
    # loaded_mat = cv2.imread('mat.bin')
        
    #############################################################    

    def close_application(self):
        QApplication.quit()
