import sys

from PyQt5.QtWidgets import QApplication

from .application import CamCalMain

__version__ = '0.0.3'

def run():
    app = QApplication(sys.argv)
    ex = CamCalMain()
    ex.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
