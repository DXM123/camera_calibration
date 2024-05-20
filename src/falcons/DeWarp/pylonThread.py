#!/usr/bin/env python
# coding: utf-8

from PyQt5.QtCore import QThread, pyqtSignal
from pypylon import pylon
import numpy as np
import cv2

# Basler Pylon : daA1920-160uc

class PylonThread(QThread):
    imageSignal = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_flag = False

    def run(self):
        #camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        tl_factory = pylon.TlFactory.GetInstance()
        camera = pylon.InstantCamera(tl_factory.GetInstance().CreateFirstDevice())

        try:
            #Settings
            camera.Open()

            # to get consistant results it is always good to start from default state
            camera.UserSetSelector.Value = "Default"
            camera.UserSetLoad.Execute()

        #try:
            CAM_HEIGHT = 800  # Example base height
            CAM_WIDTH = 608  # Example base width

            # Set width and height
            desired_width = 2 * CAM_HEIGHT
            desired_height = 2 * CAM_WIDTH
            camera.Width.SetValue(desired_width)
            camera.Height.SetValue(desired_height)

            # Set offsets
            offset_x = (1920 + 16 - 2 * CAM_HEIGHT) // 2  # center
            offset_y = (1200 + 16 - 2 * CAM_WIDTH) // 2   # center
            camera.OffsetX.SetValue(int(offset_x))
            camera.OffsetY.SetValue(int(offset_y))

            # Validate settings
            print("####### Applied Settings #########")
            print("Width set to:", camera.Width.GetValue())
            print("Height set to:", camera.Height.GetValue())
            print("OffsetX set to:", camera.OffsetX.GetValue())
            print("OffsetY set to:", camera.OffsetY.GetValue())

            # Setting camera parameters
            camera.BalanceRatioSelector.SetValue('Blue')
            camera.BalanceRatio.SetValue(1.05)
            camera.BalanceRatioSelector.SetValue('Green')
            camera.BalanceRatio.SetValue(0.75)
            camera.BalanceRatioSelector.SetValue('Red')
            camera.BalanceRatio.SetValue(0.75)

            # Validate settings
            camera.BalanceRatioSelector.SetValue('Blue')
            blue_balance = camera.BalanceRatio.GetValue()
            camera.BalanceRatioSelector.SetValue('Green')
            green_balance = camera.BalanceRatio.GetValue()
            camera.BalanceRatioSelector.SetValue('Red')
            red_balance = camera.BalanceRatio.GetValue()

            print(f"Blue Balance is:, {blue_balance}")
            print(f"Green Balance is:, {green_balance}")
            print(f"Red Balance is:, {red_balance}")

            camera.BlackLevel.SetValue(0)
            camera.BslBrightness.SetValue(0)
            camera.BslContrast.SetValue(0)

            # Validate Settings
            print(f"Black Level: {camera.BlackLevel.GetValue()}")
            print(f"Brightness: {camera.BslBrightness.GetValue()}")
            print(f"Contrast: {camera.BslContrast.GetValue()}")

            camera.ExposureMode.SetValue('Timed')
            camera.ExposureTime.SetValue(25000)

            # Validate Settings
            print(f"Exposure Mode: {camera.ExposureMode.GetValue()}")
            print(f"Exposure Time: {camera.ExposureTime.GetValue()}")

            camera.AcquisitionFrameRateEnable.SetValue(True)
            camera.AcquisitionFrameRate.SetValue(1) # try with 1 fps

            # Validate Settings
            print(f"Acquisition FrameRate Enable: {camera.AcquisitionFrameRateEnable.GetValue()}")
            print(f"Acquisition FrameRate: {camera.AcquisitionFrameRate.GetValue()}")

            camera.Gain.SetValue(11)
            camera.Gamma.SetValue(1)

            # Validate Settings
            print(f"Gain: {camera.Gain.GetValue()}")
            print(f"Gamma: {camera.Gamma.GetValue()}")

            # Set the color space to sRGB
            #camera.BslColorSpace.SetValue('BslColorSpace_sRGB') # not working
            #camera.BslColorSpace.Value = "sRGB" # not working

            camera.BslHue.SetValue(0)
            camera.BslLightSourcePreset.SetValue('Daylight5000K')
            camera.BslSaturation.SetValue(1)

            #Critical Setting
            camera.PixelFormat.SetValue('BayerRG8')

            # Validate Settings
            print(f"Hue: {camera.BslHue.GetValue()}")
            print(f"Light Source Preset: {camera.BslLightSourcePreset.GetValue()}")
            print(f"Saturation: {camera.BslSaturation.GetValue()}")

            print(f"PixelFormat: {camera.PixelFormat.GetValue()}")

            camera.TestPattern.SetValue('Off')
            camera.BalanceWhiteAuto.SetValue('Off')

            # Validate Settings
            print(f"Test Pattern: {camera.TestPattern.GetValue()}")
            print(f"Balance White Auto: {camera.BalanceWhiteAuto.GetValue()}")

            # Validate Actual achived FPS
            print("Resulting FPS:", camera.ResultingFrameRate.GetValue())

            # Set Max Buffer
            camera.MaxNumBuffer = 2 
            camera.MaxNumGrabResults = 1

            # Grabing (video) with minimal delay
            camera.StartGrabbing(1, pylon.GrabStrategy_LatestImageOnly)
            converter = pylon.ImageFormatConverter()
            
            # converting to opencv bgr format
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            #while camera.IsGrabbing():
            while camera.IsGrabbing() and not self._stop_flag:
                grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                if grabResult.GrabSucceeded():
                    image = converter.Convert(grabResult)
                    img = image.GetArray()

                    original_height, original_width = img.shape[:2]

                    # Calculate the new dimensions (half of the original dimensions)
                    new_width = original_width // 2
                    new_height = original_height // 2
                    dim = (new_width, new_height)

                    # Resize image
                    img_scaled = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

                    # Send the img instead so we can process directly in opencv
                    self.imageSignal.emit(img_scaled)

                grabResult.Release()

        except pylon.GenericException as e:
            print("An error occurred:", e)
        finally:
            if camera.IsGrabbing():
                camera.StopGrabbing()
            camera.Close()

    def stop(self):
        self._stop_flag = True
        self.wait()  # Wait for the thread to finish