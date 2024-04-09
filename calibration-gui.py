#!/usr/bin/env python3

"""
Start the dewarp calibration tool/GUI.
"""

EXAMPLE_TEXT = """Examples:
TODO?
"""


# python modules
import os
import sys
import argparse

# dewarp calibration library + GUI
import src.falcons.DeWarp


USERNAME = os.getenv('USER')



def parse_args(args: list) -> argparse.Namespace:
    """Use argparse to parse command line arguments."""
    descriptionTxt = __doc__
    exampleTxt = EXAMPLE_TEXT
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        def __init__(self, prog):
            super().__init__(prog, width=100, max_help_position=30)
    parser = argparse.ArgumentParser(description=descriptionTxt, epilog=exampleTxt, formatter_class=CustomFormatter)
    parser.add_argument('-f', '--folder', help='calibration data folder', type=str, default=default_data_folder())
    parser.add_argument('-r', '--robot', help='robot number (default prompt)', type=int)
    parser.add_argument('-c', '--camera', help='camera id (default prompt)', type=int, choices=[0,1,2,3])
    parser.add_argument('-i', '--input', help='use provided file/stream, can either be an image or a video or camera', type=str, default='/dev/video0')
    parser.add_argument('-j', '--json', help='load intermediate json file for partial calibration', type=str)
    parser.add_argument('-u', '--undistort', help='use opencv undistort instead of fisheye, intended for cameras with limited field of view', action='store_true')
    #parser.add_argument('-G', '--nogui', help='do not start the GUI, intended for regression tests', action='store_true') # maybe too much work
    return parser.parse_args(args)

# Output folder to store json, bin and captures
def default_data_folder():
    return f'/home/{USERNAME}/falcons/vision_calibration'
    # note: example structure of the suggested calibration data repo
    #     r12/20240326_195959_cam1.png      camera capture (raw, warped)
    #     r12/20240326_195959_cam1.json     readable coefficients
    #     r12/20240326_195959_cam1.bin      resulting calibration file


def ensure_prompt_int(n, message):
    if n is None:
        n = int(input(f'enter {message} (integer): '))
    return int(n)


def main(args) -> None:
    """Perform the work."""
    # prompt the robot & camera if not given
    args.robot = ensure_prompt_int(args.robot, 'robot id')
    args.camera = ensure_prompt_int(args.camera, 'camera id')
    assert args.camera in range(4)
    # configure and run the tool
    src.falcons.DeWarp.run(args)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))

