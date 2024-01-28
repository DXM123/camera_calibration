from enum import Enum


class SoccerFieldColors(Enum):
    # Define basic colors for Soccer Field (RGB)
    # Tuple[int, int, int]
    Black = (0, 0, 0)
    White = (255, 255, 255)
    Gray = (128, 128, 128)
    Red = (255, 0, 0)
    Green = (0, 255, 0)
    LightGreen = (144, 238, 144)
    DarkGreen = (0, 100, 0)
    Blue = (0, 0, 255)
    LightBlue = (173, 216, 230)
    Pink = (255, 192, 203)
    Magenta = (255, 0, 255)


class MarkerColors(Enum):
    # Define basic colors that standout on green background
    # Tuple[int, int, int]
    Yellow = (255, 255, 0)
    Orange = (255, 165, 0)
    Cyan = (0, 255, 255)
    Purple = (128, 0, 128)
    HotPink = (255, 105, 180)
    Gold = (255, 215, 0)
