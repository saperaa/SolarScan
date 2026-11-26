import numpy as np


def negative(img):
    """
    Apply negative transformation to a grayscale image.

    Input: uint8 image (0-255).
    Output: uint8 image (0-255).
    """
    return 255 - img
