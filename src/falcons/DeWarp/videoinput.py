#!/usr/bin/python3

import cv2
import numpy as np
import pyautogui # needed for screenshotting



DEFAULT_INPUT = 0 # argument for opencv VideoCapture, maps to /dev/video0



class FeedScreenShot():
    """Standard image provider to obtain a screenshot."""
    def __init__(self):
        # display screen resolution, get it using pyautogui itself
        self.resolution = tuple(pyautogui.size())
    def get(self):
        # make a screenshot
        img = pyautogui.screenshot()
        # TODO: figure out a way to (optionally) include mouse cursor? see https://stackoverflow.com/a/60266466
        # convert these pixels to a proper numpy array to work with OpenCV
        frame = np.array(img)
        return frame


class FeedVideoCapture():
    """Standard image provider to obtain a frame from any video source."""
    def __init__(self, video=0, repeat_last_when_failing=False):
        self.cap = cv2.VideoCapture(video)
        self.repeat_last_when_failing = repeat_last_when_failing
        self.frame = None # keep last frame in memory
        # determine resolution
        frame = self.get() # for example (480, 640, 3)
        self.resolution = (frame.shape[1], frame.shape[0])

    def get(self):
        ok, frame = self.cap.read()
        if ok:
            self.frame = frame # store last
            return frame
        else:
            if self.repeat_last_when_failing:
                # in case of using a video file, at end of stream: keep feeding last frame
                return self.frame
        raise Exception('could not read video frame')


class VideoInput():
    """Video input, either taken from 'screen', or something like '/dev/video1',
    or anything else acceptable by openCV VideoCapture."""
    def __init__(self, input=DEFAULT_INPUT):
        if input == 'screen':
            self.mode = 'screen'
            self.feed = FeedScreenShot()
        else:
            if input is None:
                input = DEFAULT_INPUT
            self.mode = 'video'
            self.feed = FeedVideoCapture(input, repeat_last_when_failing=True)
        self.resolution = self.feed.resolution

    def get(self):
        return self.feed.get()

    def resolution(self):
        return '{:d}x{:d}'.format(*self.resolution)

