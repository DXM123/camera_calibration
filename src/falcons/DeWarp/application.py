import json
import os
import sys

from PyQt5.QtWidgets import QAction, QFileDialog, QMainWindow, QMessageBox, QStatusBar

from .widget import CameraWidget
from .config import get_config


class CamCalMain(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "Falcons Calibration GUI - BETA"
        self.setWindowTitle(self.title)
        self.setGeometry(0, 0, 800, 600)  # #self.setGeometry(self.left, self.top, self.width, self.height)

        self.config = get_config()  # load config 
        self.check_tmp_data_empty()  # check if tmp data folder is empty

        # Initialize camera_matrix as None
        #self.camera_matrix = None

        self.init_ui()  # initialize UI

    def init_ui(self):
        # Create the central widget
        self.camera_widget = CameraWidget(self)
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
