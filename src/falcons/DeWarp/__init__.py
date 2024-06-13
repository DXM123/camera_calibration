import sys
import os


from PyQt5.QtWidgets import QApplication

from .application import CamCalMain

__version__ = '0.0.4' # TODO: this looks unused/strange, can we remove it?

def run(args):
    app = QApplication([])
    ex = CamCalMain(args.input)
    # configure based on given arguments (typically coming from argparse)
    #if not os.path.isdir(args.folder):
    #    raise Exception(f'calibration folder not found: {args.folder}')
    ex.output_folder = args.folder
    # run
    ex.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
