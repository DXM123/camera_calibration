import datetime
import json
import os
import time

import cv2
import numpy as np
import glob # for load_images
import math

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIntValidator, QDoubleValidator
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
    QGridLayout
)

from .common import MarkerColors, SoccerFieldColors
from .config import get_config
from .soccer_field import SoccerField
from .warper import Warper
from .pylonThread import PylonThread

class CalibrationInput():
    class CalibrationType:
        CAMERA = 0
        VIDEO = 1
        IMAGE = 2
        PYLON = 3

    def __init__(self):
        pass

    def getImage(self):
        pass

class CalibrationInputCamera(CalibrationInput):
    def __init__(self):
        super().__init__()

    def getImage(self):
        pass

class CalibrationInputImage(CalibrationInput):
    def __init__(self):
        super().__init__()

    def getImage(self):
        pass
class CalibrationInputNetwork(CalibrationInput):
    def __init__(self):
        super().__init__()

    def getImage(self):
        pass

class CalibrationInputPylon(CalibrationInput):
    def __init__(self):
        super().__init__()

    def getImage(self):
        pass

    

class WidgetCameraCalibration(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.inputClass = CalibrationInputCamera()
        self.initGui()

    def initGui(self):
        layout = QHBoxLayout()
        
        # display layout
        imageDisplayLayout = QVBoxLayout()
        imageDisplay = QLabel('Image Display')
        layout.addWidget(imageDisplay)


        layout.addLayout(imageDisplayLayout)


        # settings layout
        settingsLayout = QVBoxLayout()
        settingsLayout.setAlignment(Qt.AlignTop)
        settingsLayout.addWidget(QLabel('Input Options'))

        # input options
        inputTypeLayout = QHBoxLayout()
        settingsLayout.addLayout(inputTypeLayout)

        inputTypeWidget = QButtonGroup()

        inputTypeCamera = QRadioButton('Camera')
        inputTypeCamera.setChecked(True)
        inputTypeCamera.toggled.connect(lambda: self.setInputType(CalibrationInput.CalibrationType.CAMERA))
        inputTypeWidget.addButton(inputTypeCamera)
        inputTypeLayout.addWidget(inputTypeCamera)


        inputTypeImage = QRadioButton('Image')
        inputTypeImage.toggled.connect(lambda: self.setInputType(CalibrationInput.CalibrationType.IMAGE))
        inputTypeWidget.addButton(inputTypeImage)
        inputTypeLayout.addWidget(inputTypeImage)

        inputTypeNetwork = QRadioButton('Network')
        inputTypeNetwork.toggled.connect(lambda: self.setInputType(CalibrationInput.CalibrationType.NETWORK))
        inputTypeWidget.addButton(inputTypeNetwork)
        inputTypeLayout.addWidget(inputTypeNetwork)

        inputTypePylon = QRadioButton('Pylon')
        inputTypePylon.toggled.connect(lambda: self.setInputType(CalibrationInput.CalibrationType.PYLON))
        inputTypeWidget.addButton(inputTypePylon)
        inputTypeLayout.addWidget(inputTypePylon)

        # chessBoard parameters
        chessBoardLayout = QGridLayout()
        settingsLayout.addLayout(chessBoardLayout)

        chessBoardLayout.addWidget(QLabel('Colums:'), 0, 0)
        chessBoardColums = QLineEdit()
        chessBoardColums.setValidator(QIntValidator())
        chessBoardColums.setText("9")
        chessBoardLayout.addWidget(chessBoardColums, 0, 1)


        chessBoardLayout.addWidget(QLabel('Rows:'), 1, 0)
        chessBoardRows = QLineEdit()
        chessBoardRows.setValidator(QIntValidator())
        chessBoardRows.setText("6")
        chessBoardLayout.addWidget(chessBoardRows, 1, 1)
        
        chessBoardLayout.addWidget(QLabel('Square Size (mm):'), 2, 0)
        chessBoardSquareSize = QLineEdit()
        chessBoardSquareSize.setValidator(QDoubleValidator())
        chessBoardSquareSize.setText("23.0")
        chessBoardLayout.addWidget(chessBoardSquareSize, 2, 1)






        layout.addLayout(settingsLayout)

        self.setLayout(layout)

    def setInputType(self, inputType):
        print(inputType)


    def keyPressEvent(self, event):
        super().keyPressEvent(event)