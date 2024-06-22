import json
import os
import sys
import numpy as np
#import tempfile

from PyQt5.QtWidgets import QAction, QFileDialog, QMainWindow, QMessageBox, QStatusBar

from .widget import CameraWidget
from .config import get_config
from .videoinput import VideoInput


class CamCalMain(QMainWindow):
    def __init__(self, video):
        super().__init__()
        self.title = "Falcons Calibration GUI - BETA"
        self.setWindowTitle(self.title)
        #self.setGeometry(0, 0, 800, 600)  # #self.setGeometry(self.left, self.top, self.width, self.height)

        self.config = get_config()  # load config
        self.check_tmp_data_empty()  # check if tmp data folder is empty
        
        # setup tmp folder
        # by definition, a tmp folder is temporary, no interaction with user needed
        # the tmp folder gets automatically deleted when application closes
        #self.config.tmp_data = tempfile.TemporaryDirectory() # Not working since get_config() overwrites this later on
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

        # Add actions to the File menu to import LUT binary
        import_lut = QAction("Import LUT", self)  # Load LUT binary file?
        import_lut.triggered.connect(self.load_lut)
        file_menu.addAction(import_lut)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)  # of use close application function ??
        exit_action.setShortcut("Ctrl+D")
        file_menu.addAction(exit_action)

        statusbar = QStatusBar()
        self.setStatusBar(statusbar)
        statusbar.showMessage("Status: Not started")

        # Connect the signal to the slot for updating the status bar
        self.camera_widget.update_status_signal.connect(self.update_status_bar)

    def load_lut(self, filename):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        # TODO open folder to load JSON (take from args)
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select LUT binary File", "", "BIN Files (*.bin);;All Files (*)", options=options
        )

        if file_name:
            try:
                with open(file_name, "r") as f:
                    self.camera_widget.lut = self.camera_widget.load_lut_from_binary(file_name)
                    self.camera_widget.lut_imported = True


                # First, disconnect all previously connected signals to avoid multiple connections.
                try:
                    self.camera_widget.startButtonPwarp.clicked.disconnect()
                except TypeError:
                    # If no connections exist, a TypeError is raised. Pass in this case.
                    pass

                #Move to verify LUT when manually imported
                self.camera_widget.startButtonPwarp.setEnabled(True)
                self.camera_widget.startButtonPwarp.setText("Verify Lookup Table (LUT)")
                self.camera_widget.startButtonPwarp.clicked.connect(self.camera_widget.verify_lut)  # close when done

                #Switch to tab2 TODO
                # Disable the first tab (Camera Calibration)
                self.camera_widget.tabs.setTabEnabled(0, False) # Should not be here !

                # Switch to the second tab (Perspective-Warp to test LUT)
                self.camera_widget.tabs.setCurrentIndex(1)

                # Dont need this is 2e tab
                try:
                    self.camera_widget.captureButton1.clicked.disconnect()
                except TypeError:
                    # If no connections exist, a TypeError is raised. Pass in this case.
                    pass

                if self.camera_widget.imageFrame.pixmap() is None:
                    print("No Image Loaded")
                    self.camera_widget.outputWindow.setText("No Image loaded")
                    QMessageBox.question(
                        self, "Warning!", f"No Image detected.\n\nPlease make sure an Image is loaded", QMessageBox.Ok)


            except FileNotFoundError:
                print(f"File {filename} not found.")
                self.camera_widget.update_status_signal.emit(f"File not found: {file_name}")

            except Exception as e:
                print(f"Error loading calibration file: {e}")
                self.camera_widget.update_status_signal.emit("Error loading calibration file")


    def load_calibration(self, filename):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        # TODO open folder to load JSON (take from args)
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Calibration File", "", "JSON Files (*.json);;All Files (*)", options=options
        )

        if file_name:
            try:
                with open(file_name, "r") as f:
                    data = json.load(f)

                    # Convert lists to numpy arrays
                    self.camera_widget.camera_matrix = np.array(data.get("camera_matrix"))
                    self.camera_widget.camera_dist_coeff = np.array(data.get("dist_coeff"))


                    # Validate shape
                    if self.camera_widget.camera_matrix.shape != (3, 3):
                        raise ValueError("Camera matrix must be a 3x3 array.")

                    # Validate shape of distortion coefficients if necessary
                    if self.camera_widget.camera_dist_coeff.shape != (4, 1):
                        raise ValueError("Distortion coefficients must be a 4x1 array.")

                    # Emit the signal with the updated status text
                    self.camera_widget.update_status_signal.emit(f"Calibration parameters loaded from {file_name}")
                    print(f"Calibration parameters loaded from {file_name}")

                    # Debug print for camera matrix and distortion coefficients
                    print(f"Camera Matrix:\n{self.camera_widget.camera_matrix}")
                    print(f"Distortion Coefficients:\n{self.camera_widget.camera_dist_coeff}")

                    # Set tracker to True in camera_widget, needed in test_cam_calibration
                    self.camera_widget.cal_imported = True

                    # First, disconnect all previously connected signals to avoid multiple connections.
                    try:
                        self.camera_widget.doneButton1.clicked.disconnect()
                    except TypeError:
                        # If no connections exist, a TypeError is raised. Pass in this case.
                        pass

                    # Update DONE button to Test Calibration -> only when tab 1 is active TODO
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

    def check_tmp_data_empty(self):
        # Check if the temporary data folder exists for temporarily processing captured images
        if not os.path.exists(self.config.tmp_data):
            try:
                os.makedirs(self.config.tmp_data, exist_ok=True)
                QMessageBox.information(self, "Info", "The temporary data folder was created successfully.", QMessageBox.Ok)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create the temporary data folder: {e}", QMessageBox.Ok)
                sys.exit(1)  # Exits the application with an error status.
        else:
            # Check if there are existing images in the temp folder
            existing_images = [f for f in os.listdir(self.config.tmp_data) if f.startswith("corner_") and f.endswith(".png")]

            if existing_images:
                # Ask the user if they want to delete existing images
                reply = QMessageBox.question(
                    self,
                    "Existing Images",
                    "There are existing images in the temporary data folder. Do you want to delete them?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )

                if reply == QMessageBox.Yes:
                    # Delete existing images
                    for image_file in existing_images:
                        file_path = os.path.join(self.config.tmp_data, image_file)
                        os.remove(file_path)

                    # Inform the user about the deletion
                    QMessageBox.information(self, "Deletion Complete", "Existing images have been deleted.", QMessageBox.Ok)

                else:
                    # If the user chooses not to delete, inform them and exit the method
                    QMessageBox.information(self, "Calibration Canceled", "Calibration process canceled.", QMessageBox.Ok)

                    sys.exit(0) # Exits the application without an error status, this is by choice.

    def update_status_bar(self, status_text):
        # Update the status bar text
        self.statusBar().showMessage(status_text)
