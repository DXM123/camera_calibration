import json
import os
import sys
import tempfile

from PyQt5.QtWidgets import QAction, QFileDialog, QMainWindow, QMessageBox, QStatusBar

from .widget import CameraWidget
from .config import get_config
from .videoinput import VideoInput


class CamCalMain(QMainWindow):
    def __init__(self, video):
        super().__init__()
        self.title = "Falcons Calibration GUI - BETA"
        self.setWindowTitle(self.title)
        self.setGeometry(0, 0, 800, 600)  # #self.setGeometry(self.left, self.top, self.width, self.height)

        self.config = get_config()  # load config
        # setup tmp folder
        # by definition, a tmp folder is temporary, no interaction with user needed
        # the tmp folder gets automatically deleted when application closes
        self.config.tmp_data = tempfile.TemporaryDirectory()
        #print('using tmp_data ' + str(self.config.tmp_data))

        # Setup input video (or test) stream
        if video is None:
            video = 0 # default camera
        self.video_input = VideoInput(video)

        # Initialize camera_matrix as None
        #self.camera_matrix = None

        self.init_ui()  # initialize UI

    def init_ui(self):
        # Create the central widget
        self.camera_widget = CameraWidget(self, self.video_input)
        self.setCentralWidget(self.camera_widget)

        # Create the menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        # Add actions to the File menu to import calibration JSON
        import_calibration = QAction("Import Calibration", self)  # Load caibration file?
        import_calibration.triggered.connect(self.load_calibration)
        file_menu.addAction(import_calibration)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)  # of use close application function ??
        exit_action.setShortcut("Ctrl+D")
        file_menu.addAction(exit_action)

        statusbar = QStatusBar()
        self.setStatusBar(statusbar)
        statusbar.showMessage("Status: Not started")

        # Connect the signal to the slot for updating the status bar
        self.camera_widget.update_status_signal.connect(self.update_status_bar)


    def load_calibration(self, filename):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        # TODO open folder to load JSON (take from args)
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Calibration File", "", "JSON Files (*.json);;All Files (*)", options=options
        )

        if file_name:
            try:
                with open(file_name, "r") as f:
                    data = json.load(f)
                    #self.camera_matrix = data.get("camera_matrix")
                    #self.camera_dist_coeff = data.get("dist_coeff")

                    # Load into camera widget variable
                    self.camera_widget.camera_matrix = data.get("camera_matrix")
                    self.camera_widget.camera_dist_coeff = data.get("dist_coeff")

                    # Emit the signal with the updated status text
                    self.camera_widget.update_status_signal.emit(f"Calibration parameters loaded from {file_name}")
                    print(f"Calibration parameters loaded from {file_name}")

                    # Set tracker to True in camera_widget, needed in test_cam_calibration
                    self.camera_widget.cal_imported = True

                    # Update button text
                    self.camera_widget.captureButton1.setText("Capture Finished")
                    # Disable Capture Button when import succeeded
                    self.camera_widget.captureButton1.setDisabled(True)

                    # Start Camera when selected
                    if self.camera_widget.input_camera.isChecked():
                        self.camera_widget.start_capture()

                    # First, disconnect all previously connected signals to avoid multiple connections.
                    try:
                        self.camera_widget.doneButton1.clicked.disconnect()
                    except TypeError:
                        # If no connections exist, a TypeError is raised. Pass in this case.
                        pass

                    # Update DONE button to Test Calibration
                    self.camera_widget.doneButton1.setText("Test Calibration")
                    self.camera_widget.doneButton1.clicked.connect(
                        self.camera_widget.test_calibration
                    )  # change connect to calibartion test

            except FileNotFoundError:
                print(f"File {filename} not found.")
                self.camera_widget.update_status_signal.emit(f"File not found: {file_name}")

            except Exception as e:
                print(f"Error loading calibration file: {e}")
                self.camera_widget.update_status_signal.emit("Error loading calibration file")

    def update_status_bar(self, status_text):
        # Update the status bar text
        self.statusBar().showMessage(status_text)
