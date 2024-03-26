import json
import os

from PyQt5.QtWidgets import QAction, QFileDialog, QMainWindow, QMessageBox, QStatusBar

from .widget import CameraWidget


class CamCalMain(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "Falcons De-Warp Tool - BETA"
        self.setWindowTitle(self.title)
        self.setGeometry(0, 0, 800, 600)  # #self.setGeometry(self.left, self.top, self.width, self.height)

        self.output_folder = "./output"

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

                    # Set tracker to True in camera_widget, needed in test_cam_calibration
                    self.camera_widget.cal_imported = True

                    # Update button text
                    self.camera_widget.captureButton1.setText("Capture Finished")
                    # Disable Capture Button when import succeeded
                    self.camera_widget.captureButton1.setDisabled(True)

                    # Cheesy set ImageCOunter to minimum to start testing
                    # TODO What is `self.ImagesCounter` needed for?
                    # TODO Fix hard coded value below / match default of `camera_widget`?
                    self.ImagesCounter = self.camera_widget.min_cap
                    self.camera_widget.ImagesCounter = self.camera_widget.min_cap

                    # Start Camera
                    self.camera_widget.start_capture()

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
