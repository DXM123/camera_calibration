import datetime
import json
import os
import time

import cv2
import numpy as np
import glob # for load_images
import math

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

class WidgetCameraCalibration(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.initGui()
        self.parent = parent

    def initGui(self):
        pass

    def keyPressEvent(self, event):
        super().keyPressEvent(event)