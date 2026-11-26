import numpy as np
import cv2

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


def gamma_correction(img, gamma):
    """
    Apply gamma correction to a grayscale image.

    Input: uint8 image (0-255), positive gamma value.
    Output: uint8 image (0-255).
    """
    if gamma <= 0:
        raise ValueError("Gamma must be positive")

    img_float = img.astype(np.float64) / 255.0

    result = 255 * (img_float ** gamma)

    return np.clip(result, 0, 255).astype(np.uint8)


def contrast_stretch(img, low_pct, high_pct):
    """
    Apply contrast stretching using percentile limits.

    Input: uint8 image (0-255).
    Output: uint8 image (0-255).
    """
    low = np.percentile(img, low_pct)
    high = np.percentile(img, high_pct)

    if high <= low:
        return np.zeros_like(img)

    result = (img.astype(np.float64) - low) * 255 / (high - low)

    return np.clip(result, 0, 255).astype(np.uint8)


def histogram_equalization(img):
    """
    Apply histogram equalization using NumPy.

    Input: uint8 image (0-255).
    Output: uint8 image (0-255).
    """
    hist = np.bincount(img.ravel(), minlength=256)

    cdf = hist.cumsum()

    nonzero_cdf = cdf[cdf > 0]

    if len(nonzero_cdf) <= 1:
        return img.copy()

    cdf_min = nonzero_cdf[0]
    total_pixels = img.size

    lookup_table = np.round(
        255 * (cdf - cdf_min) / (total_pixels - cdf_min)
    )

    lookup_table = np.clip(lookup_table, 0, 255).astype(np.uint8)

    return lookup_table[img]


def clahe(img, clip, tiles):
    """
    Apply CLAHE to a grayscale image.

    Input: uint8 image (0-255).
    Output: uint8 image (0-255).
    """
    clahe_obj = cv2.createCLAHE(
        clipLimit=clip,
        tileGridSize=tiles
    )

    return clahe_obj.apply(img)
