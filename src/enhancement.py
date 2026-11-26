import numpy as np


def negative(img):
    """
    Apply negative transformation to a grayscale image.

    Input: uint8 image (0-255).
    Output: uint8 image (0-255).
    """
    return 255 - img

def log_transform(img):
    """
    Apply logarithmic transformation to a grayscale image.

    Input: uint8 image (0-255).
    Output: uint8 image (0-255).
    """
    img_float = img.astype(np.float64)

    c = 255 / np.log1p(255)
    result = c * np.log1p(img_float)

    return np.clip(result, 0, 255).astype(np.uint8)
