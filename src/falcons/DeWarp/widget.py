import datetime
import json
import os
import time

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
from .pylonThread import PylonThread

# MainWidget
class CameraWidget(QWidget):
    # Add a signal for updating the status
    update_status_signal = pyqtSignal(str)

    def __init__(self, parent: QMainWindow, video):
        super(QWidget, self).__init__(parent)

        self.setMouseTracking(True)  # Enable mouse tracking
        self.setFocusPolicy(Qt.StrongFocus)  # Ensure widget can accept focus

        self.video = video
        self.config = get_config()
        self.min_cap = self.config.min_cap
        self.countdown_seconds = self.config.countdown_seconds # set to initial value

        # Index for loaded images
        self.image_files = []
        self.current_image_index = -1  # Start with -1, so the first image is at index 0

        # Object points for calibration - read once from config -needs to be updated when rows or columns change
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
        self.cal_succesfull = False # Track if calibration succeeded
        self.cal_saved = False # Track if calibration is saved to file
        self.lut_imported = False # Track if LUT binary is imported
        self.lut_saved = False # Track if binairy LUT is saved to file
        self.image_dewarp = False  # Track if an image is used for dewarping
        self.video_dewarp = False  # Track if an USB camera is used for dewarping
        self.network_dewarp = False  # Track if an network stream is used for dewarping
        self.pylon_dewarp = False  # Track if an pylon stream is used for dewarping
        self.image_tuning_dewarp = False  # Track if an image tuning is used for dewarping
        self.verify_lut_started = False # Track if LUT testing is started

        #Store Final self.hv
        self.final_Hv = None

        #Store Final Points
        self.final_src_points = None

        # Need the camera object in this Widget
        self.cap = video
        self.pixmap = None

        # Add cv_image Temp (distorted frame)
        self.cv_image = None

        # Add undistorted frame
        self.undistorted_frame = None

        # Add dewarped frame
        self.dewarped_frame = None

        # Add field_image Temp
        self.field_image = None

        # Add LUT and LUT File Name
        self.lut_filename = None
        self.lut = None

        # Placeholder to count the ammount of images saved
        self.ImagesCounter = 0

        # Initialize camera_matrix as None
        self.camera_matrix = None
        self.new_K = None

        # Initialize camera_distortion coefficients as None
        self.camera_dist_coeff = None

        # Init Calibartion Frame Dimension as None
        self.calibrate_DIM = None

        # Define basic line thickness to draw soccer field
        self.line_thickness = 2

        # Set variable to store selected objectpoint for dewarp
        self.points = []
        self.selected_point = 0  # Index of the selected point for tuning dewarp

        # Also part of Warper Class
        # TODO Add logic to define landmark based on cam id [0-3]
        self.landmark_points = np.array(
            [
                self.config.field_coordinates_lm1,
                self.config.field_coordinates_lm2,
                self.config.field_coordinates_lm3,
                self.config.field_coordinates_lm4,
            ]
        )

        ############################
        # ..:: Start UI layout ::..#
        ############################

        # Use a central layout for the entire widget
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        # self.tabs.setFocusPolicy(Qt.ClickFocus)  # or Qt.StrongFocus
        #self.tabs.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # TEST TODO

        ## Prevent stealing focus -> policy! for imageFrame and CameraFrame while using Key + mouse events!! TODO

        self.tab1 = QWidget()
        self.tab2 = QWidget()
        #self.tab2.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # TEST TODO
        #self.tabs.resize(300, 200)

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
        self.optionsFrame.setFixedWidth(700) # 500 with 3 options

        # Add options widgets to optionsFrame:
        input_label = QLabel("Input Options:")
        input_label.setFixedHeight(48)
        self.optionsFrame.layout.addWidget(input_label)

        # Add radio buttons for selecting options
        self.input_layout = QHBoxLayout()

        self.input_camera = QRadioButton("Video") # Can be Camera or Video now!
        self.input_network = QRadioButton("Network")
        self.input_images = QRadioButton("Images")
        self.input_pylon = QRadioButton("Pylon")
        
        # Set "USB" as the default option
        self.input_camera.setChecked(True)

        # Connect the toggled signal of radio buttons to a slot function
        self.input_images.toggled.connect(self.update_capture_button_state) 
        self.input_network.toggled.connect(self.update_capture_button_state) # TMP - not implemented yet
        self.input_camera.toggled.connect(self.update_capture_button_state)
        self.input_pylon.toggled.connect(self.update_capture_button_state)

        # Create a button group to make sure only one option is selected at a time
        self.input_group = QButtonGroup()
        self.input_group.addButton(self.input_camera)
        self.input_group.addButton(self.input_network)
        self.input_group.addButton(self.input_images)
        self.input_group.addButton(self.input_pylon)

        self.input_layout.addWidget(self.input_camera)
        self.input_layout.addWidget(self.input_network)
        self.input_layout.addWidget(self.input_images)
        self.input_layout.addWidget(self.input_pylon)

        self.optionsFrame.layout.addLayout(self.input_layout)

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

        # Set focus on ImageFrame to receive key events -> Is this working? TODO
        #self.imageFrame.setFocusPolicy(Qt.StrongFocus)
        self.tab2inner.layout.addWidget(self.imageFrame)

        # Store the initial size for later use
        self.initialTab1InnerSize = self.tab1inner.size()
        self.initialTab2InnerSize = self.tab2inner.size()
        print(f"tab2inner initial size: {self.initialTab2InnerSize}")
        print(f"tab1inner initial size: {self.initialTab1InnerSize}")

        # Add Start De-warp Button last
        self.startButtonPwarp = QPushButton("START Selecting Landmarks", self.tab2inner)
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
        self.ProcessFrame.setFixedWidth(700) # 500 for 3 options

        # Add options widgets to optionsFrame:
        process_label = QLabel("Process Options:")
        process_label.setFixedHeight(48)
        self.ProcessFrame.layout.addWidget(process_label)

        # Add radio buttons for selecting options
        self.radio_layout = QHBoxLayout()

        #self.radio_usb = QRadioButton("USB")
        self.radio_video = QRadioButton("Video")
        self.radio_network = QRadioButton("Network")
        self.radio_image = QRadioButton("Images")
        self.radio_pylon = QRadioButton("Pylon")

        # Set "USB" as the default option
        self.radio_image.setChecked(True)

        # Connect the toggled signal of radio buttons to a slot function
        self.radio_video.toggled.connect(self.update_load_button_state)
        self.radio_network.toggled.connect(self.update_load_button_state)
        self.radio_image.toggled.connect(self.update_load_button_state)
        self.radio_pylon.toggled.connect(self.update_load_button_state)

        # Create a button group to make sure only one option is selected at a time
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.radio_video)
        self.button_group.addButton(self.radio_network)
        self.button_group.addButton(self.radio_image)
        self.button_group.addButton(self.radio_pylon)

        self.radio_layout.addWidget(self.radio_video)
        self.radio_layout.addWidget(self.radio_network)
        self.radio_layout.addWidget(self.radio_image)
        self.radio_layout.addWidget(self.radio_pylon)

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
        label_height = 400 # was 300 for 3 options

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
        elif self.input_pylon.isChecked():
            self.captureButton1.setEnabled(True)
            self.captureButton1.setText("Start Pylon Camera")
            self.captureButton1.clicked.connect(self.pylon_start)
            #self.captureButton1.clicked.connect(self.start_capture) 
        elif self.input_network.isChecked():
            self.captureButton1.setEnabled(True)
            self.captureButton1.setText("Start Capture")
            self.captureButton1.clicked.connect(self.start_capture) # Placeholder
        else:
            self.captureButton1.setEnabled(False)
            
    def pylon_start(self):
        # Using Pylon Thread instead of capture function
        #self.captureButton1.clicked.connect(self.start_capture)

        # Start Pylon Thread
        self.thread = PylonThread()

        # Connect the image signal to the slot
        #self.thread.imageSignal.connect(self.displayPylonImage) # --> Send to process function first and then update_image
        self.thread.imageSignal.connect(self.process_pylon_frame) # Get the opencv compatible img and process
        
        # Start the thread
        self.thread.start()

        # Emit the signal with the updated status text
        self.update_status_signal.emit("Pylon Camera Started...")

        # Start the countdown timer when capture is started
        self.countdown_timer.start(1000)  # Update every 1000 milliseconds (1 second)
        
        try:
            self.captureButton1.clicked.disconnect()
        except TypeError:
            # If no connections exist, a TypeError is raised. Pass in this case.
            pass

        # Make sure we can stop pylon when if needed -> TODO also when DONE is clicked (self.doneButton1)
        self.captureButton1.setText("Stop Pylon Camera")
        self.captureButton1.clicked.connect(self.pylon_stop)

    def pylon_stop(self):
        #if PylonThread:
        if PylonThread.thread:
            # Stop the thread
            self.thread.stop()

            # Stop the timer when the button is clicked again
            self.countdown_timer.stop()

            # Emit the signal with the updated status text
            self.update_status_signal.emit("Pylon Camera Stopped...")

            try:
                self.captureButton1.clicked.disconnect()
            except TypeError:
                # If no connections exist, a TypeError is raised. Pass in this case.
                pass

            #self.captureButton1.setText("Pylon Camera Stopped")
            self.captureButton1.setText("Start Pylon Camera")
            self.captureButton1.clicked.connect(self.pylon_start)

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

        # Read user-input values for columns, rows, and square size
        self.no_of_columns = int(self.columnsInput.text())
        self.no_of_rows = int(self.rowsInput.text())
        self.square_size = float(self.squareSizeRow.text())

        # Object points for calibration - re-read
        self.objp = np.zeros((1, self.no_of_columns * self.no_of_rows, 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:self.no_of_columns, 0:self.no_of_rows].T.reshape(-1, 2)

        self.current_image_index += 1
        if self.current_image_index < len(self.image_files):
            # Print which image is being loaded and its index
            print(f"Loading image {self.current_image_index + 1}/{len(self.image_files)}: {self.image_files[self.current_image_index]}")

            self.cv_image = cv2.imread(self.image_files[self.current_image_index])
            self.update_image_feed(self.cv_image)

            # Emit the signal with the updated status text
            self.update_status_signal.emit("Image Loaded...")

            # Set guiding info in output window
            self.outputWindow.setText(f"Press Enter or space to load the next image")

        else:
            print("No more images in the directory.")

            # Set guiding info in output window
            self.outputWindow.setText(f"No more images to load, press DONE to process")

            # GOTO NEXT -> Press DONE

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

            # Object points for calibration - re-read
            self.objp = np.zeros((1, self.no_of_columns * self.no_of_rows, 3), np.float32)
            self.objp[0, :, :2] = np.mgrid[0:self.no_of_columns, 0:self.no_of_rows].T.reshape(-1, 2)

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
        #img = img.rgbSwapped()  # BGR > RGB # needed for camera feed

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
            print(f"Camera Matrix used for Testing:\n{self.camera_matrix}")
            print(f"Distortion Coefficients used for Testing:\n{self.camera_dist_coeff}")

            # Generate a timestamp for the screenshot filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            filename = f"{self.config.tmp_data}/b4-cal-test_{timestamp}.png"

            # Save the frame as an image
            cv2.imwrite(filename, image)

            # Camera input not fisheye TODO create fisheye toggle -> then can remove all static references to input and
            if self.input_camera.isChecked() and self.tabs.currentIndex() == 0: # TODO This needs attention -> now only works since tab two onlyhas images
                undistorted_frame = self.undistort_frame(image, self.camera_matrix, self.camera_dist_coeff)
            else:
                undistorted_frame = self.undistort_fisheye_frame(image, self.camera_matrix, self.camera_dist_coeff)      

            # Generate a timestamp for the screenshot filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            filename = f"{self.config.tmp_data}/After-cal-test_{timestamp}.png"

            # Save the frame as an image
            cv2.imwrite(filename, undistorted_frame)

            undistorted_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
            self.pixmap = self.imageToPixmap(undistorted_frame)

            # Use cameraFrame for tab1 and imageFrame for tab2
            if self.tabs.currentIndex() == 0:  # tab1 is at index 0
                self.cameraFrame.setPixmap(self.pixmap)
                print("Tab 1 active")
            elif self.tabs.currentIndex() == 1:  # tab2 is at index 1
                self.imageFrame.setPixmap(self.pixmap)
                print("Tab 2 active")

        else:
            print("No test started or calibration file loaded")
            #org_image = self.imageToPixmap(image)

            if not self.lut_imported and not self.verify_lut_started:
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
            else:
                self.verify_lut()
                print ("Verify LUT")

                # Display the original image
                org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
                self.pixmap = self.imageToPixmap(org_image)
                self.cameraFrame.setPixmap(self.pixmap)

        # Ensure the image does not scales with the label -> issue with aspect ratio TODO
        self.cameraFrame.setScaledContents(False)
        self.update()

    def update_camera_feed(self):
        # This method will be called at regular intervals by the timer
        # - self.cap.read() instead of .get()

        # This method will be called at regular intervals by the timer
        # ret, frame = self.cap.read()
        if self.input_camera.isChecked(): # or self.radio_video
            frame = self.cap.get()
            ret = True # Why set this static when video is added !

        if self.input_pylon.isChecked() or self.input_images.isChecked():
            ret = False
            # Get frame from Thread , for now set ret to False

        #print(f"test_calibration is set to: {self.test_started}")

        if ret:  # if frame captured successfully

            # TODO Only flip for internal CAM and not input video !! -> What to do with radio_video
            if self.input_camera.isChecked() and not self.test_started:
                frame_inverted = cv2.flip(frame, 1)  # flip frame horizontally --> Is this needed TODO -> Only for Camera , not for Video !!
            else:
                frame_inverted = frame

            original_inverted_frame = frame_inverted.copy()  # Store the original frame

            # TODO update inverted_frame to corrected frame
            if self.test_started or self.cal_imported:

                # Print loaded camera matix
                #self.outputWindow.setText(f"Camera matrix used for testing:{self.camera_matrix}")

                # Camera input not fisheye TODO create fisheye toggle
                #if self.input_camera.isChecked():
                if self.input_camera.isChecked() and self.tabs.currentIndex() == 0: # TODO This needs attention -> now only works since tab two onlyhas images
                    undistorted_frame = self.undistort_frame(frame, self.camera_matrix, self.camera_dist_coeff)
                else:
                    undistorted_frame = self.undistort_fisheye_frame(frame, self.camera_matrix, self.camera_dist_coeff)
                    
                frame_inverted = undistorted_frame # cheesey replace

            if self.capture_started and not (self.test_started or self.cal_imported):
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
            else:
                # Display the original frame
                self.pixmap = self.CameraToPixmap(frame_inverted)
                self.cameraFrame.setPixmap(self.pixmap)

            # TODO Change frame_inverted name!!

            # Ensure the image does not scales with the label -> issue with aspect ratio TODO
            self.cameraFrame.setScaledContents(False)
            self.update()

    def process_pylon_frame(self, frame):
        # Read user-input values for columns, rows, and square size
        self.no_of_columns = int(self.columnsInput.text())
        self.no_of_rows = int(self.rowsInput.text())
        self.square_size = float(self.squareSizeRow.text())

        # Object points for calibration - re-read
        self.objp = np.zeros((1, self.no_of_columns * self.no_of_rows, 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:self.no_of_columns, 0:self.no_of_rows].T.reshape(-1, 2)
        
        if frame is None:
            raise ValueError("Frame is None")

        # Emit the signal with the updated status text
        self.update_status_signal.emit("Capture in progess...")

        # Get the dimensions of the original image
        original_height, original_width = frame.shape[:2]

        # Calculate the new dimensions (half of the original dimensions)
        new_width = original_width // 2
        new_height = original_height // 2
        dim = (new_width, new_height)

        # Resize image
        self.cv_image = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        org_img = self.cv_image.copy()

        # Undistort if test is started or calibration imported
        if self.test_started or self.cal_imported:

            if self.camera_matrix is None:
                raise ValueError("camera_matrix is None")

            if self.camera_dist_coeff is None:
                raise ValueError("camera_dist_coeff is None")

            # Camera input not fisheye TODO create fisheye toggle
            if self.input_camera.isChecked() and self.tabs.currentIndex() == 0: # TODO This needs attention -> now only works since tab two onlyhas images
                undistorted_frame = self.undistort_frame(self.cv_image, self.camera_matrix, self.camera_dist_coeff)
            else:
                undistorted_frame = self.undistort_fisheye_frame(self.cv_image, self.camera_matrix, self.camera_dist_coeff)       
            
            # Display the original frame
            #self.pixmap = self.CameraToPixmap(frame)
            self.pixmap = self.CameraToPixmap(undistorted_frame)
            self.cameraFrame.setPixmap(self.pixmap)
        
        else:

            detection_start_time = time.time()

            ret_corners, corners, frame_with_corners = self.detectCorners(
                self.cv_image , self.no_of_columns, self.no_of_rows
            )
            print("Corner detection took: {:.2f} seconds".format(time.time() - detection_start_time))

            if ret_corners and self.countdown_seconds > 0:
                # Optionally emit status update signal
                # self.update_status_signal.emit(f"Capturing in {self.countdown_seconds} seconds...")
                print("Capturing in", self.countdown_seconds, "seconds...")
            elif ret_corners and self.countdown_seconds == 0:
                #self.save_screenshot(frame)  # Save the original frame
                self.save_screenshot(org_img)  # Save the original resized frame
                self.countdown_seconds = self.config.countdown_seconds  # Reset the countdown after saving

            if ret_corners:
                # Display the frame with corners
                self.pixmap = self.CameraToPixmap(frame_with_corners)
                self.cameraFrame.setPixmap(self.pixmap)
                # Optional: Resize if needed
                # self.cameraFrame.resize(self.pixmap.size())
            else:
                # Display the original frame
                #self.pixmap = self.CameraToPixmap(frame)
                self.pixmap = self.CameraToPixmap(self.cv_image)
                self.cameraFrame.setPixmap(self.pixmap)
                # Optional: Resize if needed
                # self.cameraFrame.resize(self.pixmap.size())

    def displayPylonImage(self, img):
            self.pixmap = self.CameraToPixmap(img)
            self.cameraFrame.setPixmap(self.pixmap)


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
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # cv2.CALIB_CB_FILTER_QUADS really helps
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS

        # Convert to gray for better edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners. If desired number of corners are found in the image then ret = true
        #findchessboard_start_time = time.time()
        ret, corners = cv2.findChessboardCorners(
            gray,
            (columns, rows),
            flags,
        )
        #print("Find chessboard on image took: {:.2f} seconds".format(time.time() - findchessboard_start_time))

        if ret:
            print("Corners detected successfully!")
            # Now Store Object Points 
            self.object_points.append(self.objp) # Fixed

            # Refining pixel coordinates for given 2d points. A larger search window means the algorithm considers a broader area around each corner for refining its location. 
            corners_refined = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria) #-> refining on gray img for better results breaks the results!!!

            ## Now Store Corners Detected
            #self.image_points.append(corners)
            self.image_points.append(corners_refined)

            # draw and display the chessboard corners
            cv2.drawChessboardCorners(image, (columns, rows), corners_refined, ret)
        
            # Print the number of corners found
            print("Number of corners found:", len(corners_refined))

            corners = corners_refined

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

            #self.timer_pylon.stop() # Stop Pylon counter
            if self.input_pylon.isChecked():
                if PylonThread.thread:
                    self.pylon_stop() # Stop Pylon Thread

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

                    # Display the x and y coordinates of each corner
                    for i, corner in enumerate(corners):
                        x, y = corner.ravel() # Converts the corner's array to a flat array and then unpacks the x and y values
                        #print(f"Found Corner {i+1}: x = {x}, y = {y} in {file_path}")

                    # TODO Collect the Corners to be saved to corners.vnl

                    # Set succesfull calibration state
                    self.cal_succesfull = True

                    print("Calibration Succesfull")
                else:
                    print(f"No Corners Detected in {file_path}")

        # check with min_cap for minimum images needed for calibration (Default should be 10) (3 for now)
        if len(self.object_points) >= self.config.min_cap:
            print(f"Found {len(self.object_points)} images with corners detected")
            # Generate a timestamp for the screenshot filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Using frame.shape[:2][::-1] should fix the issue
            ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(
                self.object_points, self.image_points, frame.shape[:2][::-1], None, None
            )

            # TODO : frame.shape[1] vs frame.shape[::-1] vs frame.shape[:2][::-1] 

            #frame.shape[1] gives the width of the frame.
            #frame.shape[::-1] gives the dimensions of the frame in reverse order.
            #frame.shape[:2][::-1] gives the height and width of the frame in reverse order, excluding any additional dimensions like color channels.  

            if ret:  # if calibration was successfully
                # Display the calibration results
                self.outputWindow.setText(f"Camera matrix:{camera_matrix}")
                print(f"Camera matrix found:{camera_matrix}")

                # Print the RMS re-projection error
                print(f"RMS re-projection error: {ret}")

                # Check if the Root Mean Square (RMS) error is below threshold (Stored in ret)
                # Hard to get RMS below 1.0 for normal calibration               
                self.evaluate_calibration(ret)

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


    def perform_calibration_fisheye(self):

        print(f"test_calibration is set to: {self.test_started}")
        print("Start Calibration")
        self.update_status_signal.emit("Calibration in progress...")

        # At this stage object_points and image_points are alread available when detect_corners was ran for loading images / frames etc
        # Clear self.object_points and self.image_points to detect again?
        self.object_points = []  # 3D points in real world space
        self.image_points = []   # 2D points in image plane

        # Read user-input values for columns, rows, and square size once
        self.no_of_columns = int(self.columnsInput.text())
        self.no_of_rows = int(self.rowsInput.text())
        self.square_size = float(self.squareSizeRow.text())

        # Object points for calibration - re-read
        self.objp = np.zeros((1, self.no_of_columns * self.no_of_rows, 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:self.no_of_columns, 0:self.no_of_rows].T.reshape(-1, 2)

        # Now using tmp folder instead of input folder
        image_files = sorted(os.listdir(self.config.tmp_data))
        
        for file_name in image_files:
            if file_name.startswith("corner_") and file_name.endswith(".png"):
                file_path = os.path.join(self.config.tmp_data, file_name)
                frame = cv2.imread(file_path)

                
                ret_corners, corners, _ = self.detectCorners(frame, self.no_of_columns, self.no_of_rows)

                if ret_corners:

                    # Display the x and y coordinates of each corner
                    #for i, corner in enumerate(corners):
                    #    x, y = corner.ravel() # Converts the corner's array to a flat array and then unpacks the x and y values
                        #print(f"Found Corner {i+1}: x = {x}, y = {y} in {file_path}")

                    # TODO Collect the Corners to be saved to corners.vnl

                    print("Corner Detection Succesfull")

        if len(self.object_points) >= self.config.min_cap:  # Ensure there's enough data for calibration
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            obj_points = np.array(self.object_points)
            img_points = np.array(self.image_points)
            #_img_shape = frame.shape[:2]
            _img_shape = frame.shape[:2][::-1]

            # Print shapes of inputs
            print(f"Object Points Shape: {obj_points.shape}")
            print(f"Image Points Shape: {img_points.shape}")
            
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
                    #_img_shape[::-1],
                    _img_shape,
                    K,
                    D,
                    rvecs,
                    tvecs,
                    calibration_flags,
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                )

                # Check if the Root Mean Square (RMS) error is below threshold                    
                self.evaluate_calibration(rms)

                # Set succesfull calibration state
                self.cal_succesfull = True

                self.outputWindow.setText(f"Camera matrix found:{K}\nDistortion coefficients found:{D}")
                print(f"Camera matrix found:{K}\nDistortion coefficients found:{D}")

                self.camera_matrix = K
                self.camera_dist_coeff = D

                #self.calibrate_DIM = _img_shape[::-1]
                self.calibrate_DIM = _img_shape

                # Save intrinsic parameters to intrinsic.txt
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


            except cv2.error as e:
                print(f"Calibration failed with error: {e}")

        else:
            print(f"Need at least {self.config.min_cap} images with corners for calibration. Only {len(self.object_points)} found")


    def verify_rms_prev(self, rms):
        # Exceptional: RMS < 0.1
        if rms < 0.1:
            return "Exceptional"

        # Very Good: 0.1 <= RMS < 0.3
        elif rms < 0.3:
            return "Very Good"

        # Good: 0.3 <= RMS < 0.5
        elif rms < 0.5:
            return "Good"

        # Acceptable: 0.5 <= RMS < 1.0
        elif rms < 1.0:
            return "Acceptable"

        # Acceptable: 0.5 <= RMS < 1.0
        elif rms > 1.0:
            return "Not Acceptable"

        # Poor: RMS >= 1.0
        else:
            return "Very Poor"

    def evaluate_calibration_prev(self, rms):
        # Check RMS and categorize calibration quality (Work for Fisheye, but diff valies for non fisheye)
        quality = self.verify_rms(rms)

        # Print message or raise error based on RMS quality
        if quality in ["Exceptional", "Very Good", "Good"]:
            #print(f"RMS is {rms}, indicating {quality} calibration quality.")
            green_bold = "\033[32m\033[1m"
            reset = "\033[0m"  # Resets the style to default
            print(f"{green_bold}RMS is {rms}, indicating {quality} calibration quality.{reset}")
        elif quality is ["Acceptable"]:
            orange_bold = "\033[38;2;255;165;0m\033[1m"
            reset = "\033[0m"  # Resets the style to default
            print(f"{orange_bold}RMS is {rms}, indicating {quality} calibration quality.{reset}")
        else:
            self.timer.stop()
            raise ValueError(f"RMS is {rms}, indicating Poor calibration quality. Recalibration required.")
            # Add Retry option

    def verify_rms(self, rms):
        # Exceptional: RMS < 0.1
        if rms < 0.1:
            return "Exceptional"

        # Very Good: 0.1 <= RMS < 0.3
        elif rms < 0.3:
            return "Very Good"

        # Good: 0.3 <= RMS < 0.5
        elif rms < 0.5:
            return "Good"

        # Acceptable: 0.5 <= RMS < 1.0
        elif rms < 1.0:
            return "Acceptable"

        # Not Acceptable: RMS >= 1.0
        else:
            return "Not Acceptable"

    def evaluate_calibration(self, rms):
        # Check RMS and categorize calibration quality
        quality = self.verify_rms(rms)

        # Print message or raise error based on RMS quality
        if quality in ["Exceptional", "Very Good", "Good"]:
            green_bold = "\033[32m\033[1m"
            reset = "\033[0m"  # Resets the style to default
            print(f"{green_bold}RMS is {rms}, indicating {quality} calibration quality.{reset}")
        elif quality == "Acceptable":
            orange_bold = "\033[38;2;255;165;0m\033[1m"
            reset = "\033[0m"  # Resets the style to default
            print(f"{orange_bold}RMS is {rms}, indicating {quality} calibration quality.{reset}")
        else:
            self.timer.stop()
            raise ValueError(f"RMS is {rms}, indicating Poor calibration quality. Recalibration required.")


    def generate_object_points(self, columns, rows, square_size):
        objp = np.zeros((1, columns * rows, 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size
        return objp

    # always in tab 1
    def test_calibration(self):
        # TODO if Pause button was pressed , camera stops !!!!
        print("Testing Calibration")

        # Set test boolean to prevent mutiple saves (or 10 rows lower?)
        self.test_started = True

        # Start Calibration again also allow it to save intrinsic / extrinsic tmp files once -> Not working due to test_started value = True
        if self.input_images.isChecked():
            if not self.cal_imported:
                self.perform_calibration_fisheye()
                #self.perform_calibration()
                self.current_image_index = -1
                self.load_next_image() #

        if self.input_camera.isChecked():
            if not self.cal_imported:
                self.perform_calibration()
            self.timer.start(100) # Start camera feed

########################################################################################

        if self.input_pylon.isChecked():
            if not self.cal_imported:
                self.perform_calibration_fisheye() # Somethings off using Pylon and the <New Camera Matrix:> calculated
            self.pylon_start()

#######################################################################################

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
            
            #DIM = self.calibrate_DIM

            # Optimize Matrix
            #h, w = frame.shape[:2]
            dim = frame.shape[:2][::-1]

            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, dim, 1, dim)

            # Undistort the frame using the camera matrix and distortion coefficients
            #undistorted_frame = cv2.undistort(frame, camera_matrix, distortion_coefficients)
            undistorted_frame = cv2.undistort(frame, camera_matrix, distortion_coefficients, None, new_camera_matrix)

            return undistorted_frame

        else:
            print("No camera matrix or distortion coefficient detected, showing original frame")

            return frame
        
    # When lens field of view is above 160 degree we need fisheye undistort function for opencv (This one works, but cuts to much of the frame)

    # Play with Balance / ROI ! What is required for landmark selection !!! balance=0 is ROI vs balance=1 whole picture undistorted, or 0.8 for original :)

    def undistort_fisheye_frame(self, frame, camera_matrix, distortion_coefficients, balance=0, dim2=None, dim3=None):

        if camera_matrix is None:
            raise ValueError("camera_matrix is None")
        if frame is None:
            raise ValueError("cv_image is None")
        if distortion_coefficients is None:
            raise ValueError("camera_dist_coeff is None")


        if camera_matrix is None or distortion_coefficients is None:
            print("No camera matrix or distortion coefficient detected, showing original frame")
            return frame

        DIM = self.calibrate_DIM # Usue when inporting calibration , then this is None TODO Also store DIM in Calibration JSON
        print(f"Frame Dimension (DIM) = {DIM}")

        # Below working for pylon / cam (not for images)
        dim1 = frame.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
        # TODO : frame.shape[1] vs frame.shape[::-1] vs frame.shape[:2][::-1] 

        # Temp work a round when importing calibration
        if not DIM and self.cal_imported:
            DIM = dim1

        #frame.shape[1] gives the width of the frame.
        #frame.shape[::-1] gives the dimensions of the frame in reverse order.
        #frame.shape[:2][::-1] gives the height and width of the frame in reverse order, excluding any additional dimensions like color channels.

        #h, w = frame.shape[:2]
        print(f"Frame Dimension input (dim1) = {dim1}")

        if not dim2:
            dim2 = dim1 # Target dimension for the new camera matrix optimization
        if not dim3:
            dim3 = dim1 # Output image dimension for undistortion map

        scaled_K = camera_matrix * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

        # Now, estimate the new camera matrix optimized for dim2 dimensions
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, distortion_coefficients, dim2, np.eye(3), balance=balance)
        self.new_K = new_K
        
        #print(f"Original Camera Matrix:\n{camera_matrix}")
        #print(f"New Camera Matrix:\n{new_K}")

        # Initialize the undistort rectify map for the dimensions of the output image (dim3)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, distortion_coefficients, np.eye(3), new_K, dim3, cv2.CV_16SC2)

        # Finally, remap the image using the undistortion map for the corrected image
        undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) # verify interpolation

        return undistorted_frame


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

            # Stop any conuntdowns
            self.countdown_timer.stop()

            if self.input_pylon.isChecked():
                if PylonThread.thread:
                    self.pylon_stop()

            # Stop Input
            self.cameraFrame.keyPressEvent = None
            self.cameraFrame.mousePressEvent = None
            self.cameraFrame.releaseMouse()
            self.cameraFrame.releaseKeyboard()

        except Exception as e:
            # Handle any exceptions that may occur during file writing
            print(f"Error saving calibration file: {e}")
            self.update_status_signal.emit("Error saving calibration file")

    # Below is mostly tab2 related to Perspective-Warp

    # Slot function to enable or disable the "Load Image" button based on the radio button state
    def update_load_button_state(self):
        if self.radio_video.isChecked() or self.radio_network.isChecked():
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
            
            print(f"Loading file_name type: {type(file_name)}, file_name value: {file_name}")

            self.display_image(file_name)

    def start_pwarp(self):
        # Stop Camera -> also stopped after save
        #self.timer.stop() # Should not be here ! works for now since only images are used

        if self.cal_imported == True:
            # Set image Dewarp to True
            self.image_dewarp = True  # TODO Does not belong here

            print(f"Calibration imported is: {self.cal_imported}")

        if self.cal_succesfull == True:
            # Set image Dewarp to True
            self.image_dewarp = True  # TODO Does not belong here

            print(f"Calibration imported used: {self.cal_imported}")

        # Check if the pixmap is set on the Image Frane
        if self.imageFrame.pixmap() is not None:

            # Temp disable Start button untill all 4 points are collected TODO
            self.startButtonPwarp.setDisabled(True)  # Should not be here !

            if self.image_dewarp == True:
                print("Image Perspective Warp started")
                self.update_status_signal.emit("Image Perspective Warp started")

                # Get frame --> TODO check all placed where self.cv_image is stored
                frame = self.cv_image

                # Camera input not fisheye TODO create fisheye toggle
                if self.input_camera.isChecked() and self.tabs.currentIndex() == 0: # TODO This needs attention -> now only works since tab two onlyhas images
                #if self.input_camera.isChecked():
                    undistorted_frame = self.undistort_frame(frame, self.camera_matrix, self.camera_dist_coeff)
                else:
                    undistorted_frame = self.undistort_fisheye_frame(frame, self.camera_matrix, self.camera_dist_coeff)

                undistorted_frame_rgb = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
                self.pixmap = self.imageToPixmap(undistorted_frame_rgb)

                # Check if 'frame' is a valid NumPy array
                if isinstance(frame, np.ndarray):
                    # Disable Load image Button when import succeeded and dewarp started
                    self.loadImage.setDisabled(True)  # LoadImage / load_image is confusing TODO

                    if len(frame.shape) == 3:  # Verify if valid frame shape
                        
                        self.warper_result = self.dewarp(undistorted_frame_rgb)  # return warper                     

                        # Perform dewarping
                        self.dewarped_frame = self.warper_result.warp(undistorted_frame_rgb.copy(),self.field_image.copy())

                        self.display_frame(self.dewarped_frame)

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

                    else:
                        print("Invalid frame format: Not a 3D array (color image)")
                else:
                    print("Invalid frame format: Not a NumPy array")

            elif self.video_dewarp == True:
                # video_dewarp()
                print("Starting Video de-warp")  # -> update status

                # Start the camera again
                self.timer.start(100)  # Assuming 100 milliseconds per frame update

                # TODO
                #ret, frame = self.cap.read()  # read frame from webcam   -> use update_camera function
                frame = self.cap.get()
                ret = True

            elif self.network_dewarp == True:
                # network_dewarp()
                print("Starting network de-warp")  # -> update status

            ##############################################################

            # Start Pylon Thread again
            elif self.pylon_dewarp == True:
                # network_dewarp()
                print("Starting Pylon de-warp")  # -> update status

            ############################################################3

        else:
            # Disable the first tab (Camera Calibration)
            #self.tabs.setTabEnabled(0, False) # Should not be here !

            # Clear Image from self.imageFrame
            self.cameraFrame.setPixmap(QPixmap())
            self.imageFrame.setPixmap(QPixmap()) 

            # Switch to the second tab (Perspective-Warp)
            self.tabs.setCurrentIndex(1) # Should not be here !

            # Dont need this is 2e tab
            try:
                self.captureButton1.clicked.disconnect()
                self.cameraFrame.keyPressEvent = None
                self.cameraFrame.releaseKeyboard()
                self.tabs.setTabEnabled(0, False) # Should not be here !
            except TypeError:
                # If no connections exist, a TypeError is raised. Pass in this case.
                pass

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

            # Start collecting landmarks with mouse clicks - Add zoom with Z at landmark TODO
            self.imageFrame.mousePressEvent = self.mouse_click_landmark_event

        if self.image_tuning_dewarp == True:
            print("Perspective Tuning is started")

            # Stop mouse and key press event registration
            # self.imageFrame.mousePressEvent = None
            # self.imageFrame.keyPressEvent = None

            # Start collecting arrow key events
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

                if not self.image_tuning_dewarp ==True and not self.verify_lut_started:
                    # Stop mouse press event registration
                    print("Disabeling mouse input on image Frame")
                    self.imageFrame.mousePressEvent = None
                    self.imageFrame.releaseMouse()
                    # self.imageFrame.keyPressEvent = None
                elif self.verify_lut_started:
                    print("Enabeling mouse input on image Frame")
                    self.imageFrame.mousePressEvent = self.mouse_click_landmark_event

                if self.image_tuning_dewarp:

                    self.startButtonPwarp.setText("Click when done tuning")   ## Not updating TODO
                    self.startButtonPwarp.clicked.connect(self.stop_tuning)

                break

            QApplication.processEvents()

        print(
            f"Check Widget UI Frame | W: {self.width()}, H: {self.height()}\n"
            f"Check imageFrame| W: {self.imageFrame.width()}, H: {self.imageFrame.height()}\n"
            f"Check image Shape| W: {img.shape[0]}, H: {img.shape[1]}\n"
        )

        # Test is value is provided
        #print(f"Pre-Warper Camera Distortion Coefficients: {self.camera_dist_coeff}")
        #print(f"Pre-Warper Camera Matrix: {self.camera_matrix}")

        # TODO Can we fix this sorting without hard-coding to the config?
        self.warper = Warper(
            points=np.array([self.points[0], self.points[1], self.points[2], self.points[3]]),
            landmark_points=self.landmark_points,
            width=img.shape[1],
            height=img.shape[0],
            matrix=self.camera_matrix,
            new_matrix=self.new_K,
            dist_coeff=self.camera_dist_coeff,
        )            

        return self.warper 

    def stop_tuning(self):
        self.image_tuning_dewarp == False

        #Final self.hv
        print(f"Final Homography Matrix (Hv) after tweaking: {self.warper.Hv}")
        self.final_Hv = self.warper.Hv

        #Final Points
        print(f"Final src points after tweaking: {self.warper.src_points}")
        self.final_src_points = self.warper.src_points

        # Disable input
        #self.imageFrame.mousePressEvent = None
        #self.imageFrame.keyPressEvent = None

        # Disable widget -> Maybe a bit much
        self.imageFrame.setEnabled(False)

        # First, disconnect all previously connected signals to avoid multiple connections.
        try:
            self.startButtonPwarp.clicked.disconnect()
        except TypeError:
            # If no connections exist, a TypeError is raised. Pass in this case.
            pass

        # Set save options TODO
        self.startButtonPwarp.setText("Save to binary file")
        self.startButtonPwarp.clicked.connect(self.save_prep_mat_binary)

        ###############################################################
        # Move to Testing LUT generated by save_prep_mat_binary

        if self.lut_saved == True:

            # First, disconnect all previously connected signals to avoid multiple connections.
            try:
                self.startButtonPwarp.clicked.disconnect()
            except TypeError:
                # If no connections exist, a TypeError is raised. Pass in this case.
                pass

            #self.startButtonPwarp.setText("DONE")
            self.startButtonPwarp.setText("Verify Lookup Table (LUT)")
            self.startButtonPwarp.clicked.connect(self.verify_lut)  # close when done

#########################################################################################

    def verify_lut(self):
        print(f"Start Verify Lookup Table (LUT)")
        # CLick (Mouse Click event) ImageFrame and get x,y -> get x,y from LUT and show transformed point of imageField
        # Input Generated LUT binary file
        # Input Mouse Clicks x,y on imageFrame
        # OutPut x,y on field_image

        try:
            # Stop mouse and key press event registration
            # self.imageFrame.mousePressEvent = None
            self.imageFrame.keyPressEvent = None
            self.imageFrame.releaseKeyboard()
            self.releaseKeyboard()
        except TypeError:
            # If no connections exist, a TypeError is raised. Pass in this case.
            pass

        ######################################

        # Set toggle to false to prevent undistort image -> but verify works ok on undistorted image ??
        self.test_started = False
        self.cal_imported = False

        #######################################

        # Only load LUT once during start
        if not self.verify_lut_started:
            print("LUT verification not started. Initializing LUT...")
            ### Could also load lut like:
            #self.lut = None # Clear if needed
            if self.lut is None:
                print("LUT is None. Loading LUT from binary file...")
                #Load LUT 
                self.lut = self.load_lut_from_binary(self.lut_filename)
                print(f"LUT loaded successfully from {self.lut_filename}")
                
                # now must be mat.bin in root TODO add way to load it like calibration in 3e tab?
                # Not supported now to load a previous generated LUT

            # Set LUT verify as Started
            self.verify_lut_started = True
            print("LUT verification started.")
            #self.imageFrame.setEnabled(True) # Enable ImageFrame again
            #self.imageFrame.mousePressEvent = self.mouse_click_landmark_event
            self.verify_lut()
            print ("Verify LUT")
        else:
            print("LUT verification already started. Skipping initialization.")
            self.imageFrame.setEnabled(True) # Enable ImageFrame again
            #self.loadImage.setEnabled(True) # Enable load image
            print("Enabeling mouse input on image Frame")

            # Enable mouse click events
            self.imageFrame.mousePressEvent = self.mouse_click_landmark_event

        self.loadImage.setEnabled(True) # Enable load image

        # First, disconnect all previously connected signals to avoid multiple connections.
        try:
            self.startButtonPwarp.clicked.disconnect()
        except TypeError:
            # If no connections exist, a TypeError is raised. Pass in this case.
            pass

        # For now Quit App
        self.startButtonPwarp.setText("DONE")
        self.startButtonPwarp.clicked.connect(
            self.close_application
        ) 

######################################################################################

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
        cv2.circle(image, landmark, 30, (color), 5)

        return image

    # Image loading for tab2 - image
    def display_frame(self, dewarped_frame):
        # Display the dewarped image
        dewarped_pixmap = self.imageToPixmap(dewarped_frame)
        self.imageFrame.setPixmap(dewarped_pixmap)
        self.imageFrame.setScaledContents(True)

        
    # Image file loading for tab2 - filename
    def display_image(self, file_name):

        print(f"Filename is: {file_name}")

        if file_name:
            # Image loading and processing logic here
            self.cv_image = cv2.imread(file_name)
            self.cv_image_rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)  # Corrected color conversion

            # Covert to Pixmap and other operations...
            pixmap = self.imageToPixmap(self.cv_image_rgb)

            # Set loaded image in ImageFrame
            self.imageFrame.setPixmap(pixmap)
            self.imageFrame.setScaledContents(False) # Never Scale

            # Adjust imageFrame size to match the pixmap size - Set Fixed Size like imageFrame
            self.imageFrame.setFixedSize(pixmap.size())

            # Allign Buttons to the same size self.startButtonPwarp & self.loadImage
            # Get the width of cameraFrame
            imageFrameWidth = self.imageFrame.size().width()

            # Set the buttons to the same width as cameraFrame
            self.doneButton1.setFixedWidth(imageFrameWidth)
            self.captureButton1.setFixedWidth(imageFrameWidth)

            if self.imageFrame.pixmap() is not None:
                self.processoutputWindow.setText("Image loaded")
                    
                self.update_image_feed(self.cv_image) 

            else:
                self.processoutputWindow.setText("Problem displaying image")

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

            # Debug Mouse events
            print(f"Clicked at local coordinates: ({event.x()}, {event.y()})")
            globalPos = self.mapToGlobal(event.pos())
            print(f"Clicked at global coordinates: ({globalPos.x()}, {globalPos.y()})")
            super().mousePressEvent(event)  # Call the parent class's event handler

            # Check if the pixmap is set on the Image Frane
            if self.imageFrame.pixmap() is not None:
                self.points.append((x, y))
                self.imageFrame.setPixmap(QPixmap())  # Clear the current pixmap -> Might cause flickering

                frame = self.cv_image

                if self.test_started or self.cal_imported:

                    # Camera input not fisheye TODO create fisheye toggle
                    #if self.input_camera.isChecked():
                    if self.input_camera.isChecked() and self.tabs.currentIndex() == 0: # TODO This needs attention -> now only works since tab two onlyhas images
                        undistorted_frame = self.undistort_frame(frame, self.camera_matrix, self.camera_dist_coeff)
                    else:
                        undistorted_frame = self.undistort_fisheye_frame(frame, self.camera_matrix, self.camera_dist_coeff)

                    undistorted_frame_rgb = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
                else:
                    # TODO Update name
                    undistorted_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                bgimage = cv2.rectangle(undistorted_frame_rgb, (x, y), (x + 2, y + 2), (SoccerFieldColors.Green.value), 2)
                self.display_landmarks(bgimage)

                if self.verify_lut_started:
                    transformed_point = self.query_lut(self.lut, x, y)

                    if transformed_point:
                        print(f"Original point: ({x}, {y})")
                        print(f"Transformed point: {transformed_point}")

                        #Refine transformed point to support x, y
                        # Round the transformed point coordinates to the nearest integers
                        rounded_transformed_point = (round(transformed_point[0]), round(transformed_point[1]))

                        print(f"Transformed point (Rounded): {rounded_transformed_point}")

                        # Draw Transformed points on soccer Field Image
                        #self.field_image_selected = self.draw_landmark_selected(
                        #    self.field_image.copy(), transformed_point, MarkerColors.Yellow.value
                        #)

                        # Using rounded values
                        self.field_image_selected = self.draw_landmark_selected(
                            self.field_image.copy(), rounded_transformed_point, MarkerColors.Yellow.value
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
                        print("Point is out of the bounds of the LUT.")

                ####################################################################
                # Draw Transformed points on soccer Field Image
                #self.field_image_selected = self.draw_landmark_selected(
                #    self.field_image.copy(), self.landmark_points[self.selected_point], MarkerColors.Yellow.value
                #)
                ##################################################################

            else:
                print("Pixmap is not set")

    # Overriding focusInEvent and focusOutEvent helps debug whether the widget has gained or lost focus.
    def focusInEvent(self, event):
        # Debug focus events
        print("Widget has gained focus")
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        # Debug focus events
        print("Widget has lost focus")
        super().focusOutEvent(event)

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
                    
                # Cleanup - dont want to store previous landmarks (or dont overwrite :D)
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

            # Add Zoom in: Resize the pixmap to 300% of its original size
            # How to use self.dewarped_frame via ImageFrame? TODO
            #elif event.key() == Qt.Key_Z:  # Check if 'Z' was pressed

            else:
                super().keyPressEvent(event)  # Pass other key events to the base class

            # Make rectangle red when selected
            print(f"Updating perspective transform, using: self.points (pts{self.selected_point}): {self.points}")

            #When validating LUT , no undistort
            if self.test_started or self.cal_imported:

                # Check if the pixmap is set on the Image Frane
                if self.imageFrame.pixmap() is not None:
                    print("Update Image Frame / PWarp view")

                    # Debug print for camera matrix and distortion coefficients
                    print(f"Camera Matrix:\n{self.camera_matrix}")
                    print(f"Distortion Coefficients:\n{self.camera_dist_coeff}")

                    frame = self.cv_image

                    # Camera input not fisheye TODO create fisheye toggle
                    #if self.input_camera.isChecked():
                    if self.input_camera.isChecked() and self.tabs.currentIndex() == 0: # TODO This needs attention -> now only works since tab two only has images
                        undistorted_frame = self.undistort_frame(frame, self.camera_matrix, self.camera_dist_coeff)
                    else:
                        undistorted_frame = self.undistort_fisheye_frame(frame, self.camera_matrix, self.camera_dist_coeff)

                    undistorted_frame_rgb = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
                    self.pixmap = self.imageToPixmap(undistorted_frame_rgb)

                    # Check if 'frame' is a valid NumPy array
                    if isinstance(frame, np.ndarray):
                        
                        # Disable Load image Button when import succeeded and dewarp started
                        self.loadImage.setDisabled(True)  # LoadImage / load_image is confusing TODO

                        if len(frame.shape) == 3:  # Verify if valid frame shape

                            self.warper_result = self.dewarp(undistorted_frame_rgb)  # return warper                        

                            # Perform dewarping
                            self.dewarped_frame = self.warper_result.warp(undistorted_frame_rgb.copy(),self.field_image.copy())

                            # Add functionality while tuning
                            if self.image_tuning_dewarp == True:
                                # Draw selector on dewarped frame
                                self.dewarped_frame = self.draw_landmark_selected(
                                    self.dewarped_frame.copy(), self.landmark_points[self.selected_point], MarkerColors.Yellow.value
                                )

                            self.display_frame(self.dewarped_frame)

                            # Print stuff and update status bar
                            print("Dewarping process completed.")
                            self.processoutputWindow.setText("Dewarping process completed.")
                            self.update_status_signal.emit("Dewarping process completed.")

                else:
                    print("Pixmap is not set")
            else:
                frame = self.cv_image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.pixmap = self.imageToPixmap(frame_rgb)


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

    def tweak_pwarp(self):
        # Print new instruction
        # self.processoutputWindow.setText("Tuning Landmarks started")

        # Add check if Tuning is started
        self.image_tuning_dewarp = True  # Track if an image tuning is used for dewarping

        # Start de warp again for tuning
        self.start_pwarp()

    ####################### Binary file stuff WIP ###################
    
    def write_mat_binary(self, ofs, out_mat):
        if not ofs:
            return False
        if out_mat is None:
            s = 0
            ofs.write(s.to_bytes(4, byteorder='little'))
            return True
        rows, cols = out_mat.shape[:2]
        dtype = out_mat.dtype.itemsize
        ofs.write(rows.to_bytes(4, byteorder='little'))
        ofs.write(cols.to_bytes(4, byteorder='little'))
        ofs.write(np.uint32(dtype).tobytes())
        ofs.write(out_mat.tobytes())

        print(f"Binary LUT saved to {self.lut_filename}")
        self.processoutputWindow.setText(f"Binary LUT saved to {self.lut_filename}")
        self.update_status_signal.emit(f"Binary LUT saved to {self.lut_filename}")

        self.lut_saved = True # Track if calibration is saved to file

        return True
    
    def save_prep_mat_binary(self):
        # Set Filename static for now TODO add date and get from config
        filename = "mat.bin"
        self.lut_filename = filename

        # Get Shape of dewarped image
        img_shape = self.dewarped_frame.shape[:2] # Do we save everything? Yes for now
        print(f"Shape of dewarped image: {img_shape}")

        # Get Shape of src image
        #src_img = (self.warper.src_height, self.warper.src_width)
        print(f"Source image dimensions: Height = {self.warper.src_height}, Width = {self.warper.src_width}")

        # Assign src_img as the new img_shape -> to make smaller LUT?
        #img_shape = src_img

        print(f"LUT shape: {img_shape}")

        print(f"Generating the binary LUT, please be patient!")

        #Below is not beeing displayed !!!
        self.processoutputWindow.setText(f"Generating the binary LUT, please be patient!")
        self.update_status_signal.emit(f"Generating the binary LUT, please be patient!")

        try:
            self.startButtonPwarp.setEnabled(False)
            # Create the lookup table
            self.lut = self.warper.create_lookup_table(img_shape)
        except TypeError:
            pass

        # Create the lookup table
        #self.lut = self.warper.create_lookup_table(img_shape)

        #return self.save_mat_binary(self.lut_filename, self.lut)
        self.save_mat_binary(self.lut_filename, self.lut)

        ###############################################################
        # Move to Testing LUT generated by save_prep_mat_binary
        if self.lut_saved == True:

            self.startButtonPwarp.setEnabled(True)

            # First, disconnect all previously connected signals to avoid multiple connections.
            try:
                self.startButtonPwarp.clicked.disconnect()
                self.imageFrame.setPixmap(QPixmap()) # clear frame
            except TypeError:
                # If no connections exist, a TypeError is raised. Pass in this case.
                pass

            #self.startButtonPwarp.setText("DONE")
            #self.startButtonPwarp.setEnabled(True)
            self.startButtonPwarp.setText("Verify Lookup Table (LUT)")
            self.startButtonPwarp.clicked.connect(self.verify_lut)  # close when done

    def save_mat_binary(self, filename, output):
        with open(filename, 'wb') as ofs:
            return self.write_mat_binary(ofs, output)
        
    def load_lut_from_binary(self, filename):

        with open(filename, 'rb') as file:
            rows = int.from_bytes(file.read(4), byteorder='little')
            cols = int.from_bytes(file.read(4), byteorder='little')
            dtype_size = int.from_bytes(file.read(4), byteorder='little')
            
            # Assuming the LUT was saved as float32 for both x' and y' values
            dtype = np.float32
            lut = np.frombuffer(file.read(), dtype=dtype).reshape((rows, cols, 2))

            print(f"LUT binary {filename} loaded")

            # Emit the signal with the updated status text
            self.update_status_signal.emit(f"LUT binary {filename} loaded")

            # Set guiding info in output window
            #self.outputWindow.setText(f"LUT binary {filename} loaded")
            #self.processoutputWindow.setText(f"LUT binary {filename} loaded")
            
        return lut

    def query_lut(self, lut, x, y):

        if 0 <= y < lut.shape[0] and 0 <= x < lut.shape[1]:
            return tuple(lut[y, x])
        else:
            return None  # Out of bounds

    def close_application(self):
        QApplication.quit()