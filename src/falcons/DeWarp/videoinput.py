#!/usr/bin/python3

import cv2
import numpy as np



DEFAULT_INPUT = 0 # argument for opencv VideoCapture, maps to /dev/video0





class FeedVideoCapture():
    """Standard image provider to obtain a frame from any video source."""
    def __init__(self, video=DEFAULT_INPUT, repeat_last_when_failing=False):
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
    def __init__(self, video_input=DEFAULT_INPUT):
        if video_input == 'screen':
            raise NotImplementedError('stripped functionality that was depending on pyautogui')
        else:
            if video_input is None:
                video_input = DEFAULT_INPUT
            self.mode = 'video'
            self.feed = FeedVideoCapture(video_input, repeat_last_when_failing=True)
        self.resolution = self.feed.resolution

    def get(self):
        return self.feed.get()

    def resolution(self):
        return '{:d}x{:d}'.format(*self.resolution)

