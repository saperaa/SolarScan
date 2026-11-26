"""Thresholding and segmentation of dark defect candidates.

Owner: Segmentation Engineer (role 5, Mohamed Alaa).

Defects in EL images are darker than the active cell, so every mask returned here is a
``bool`` (H, W) array where True marks a dark pixel (a defect candidate). Inputs are
``uint8`` (H, W) grayscale images with values 0-255.

NumPy implementations (checked against libraries in ``tests/test_segmentation.py``):
    - ``otsu_threshold``      vs ``skimage.filters.threshold_otsu`` and OpenCV's Otsu
    - ``adaptive_threshold``  vs ``cv2.adaptiveThreshold`` (mean method)
"""

import numpy as np


def _check_image(image: np.ndarray) -> None:
    """Require a non-empty ``uint8`` (H, W) array."""
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise ValueError("image must be a non-empty 2-D array")
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8 (values 0-255)")


def otsu_threshold(image: np.ndarray) -> int:
    """Otsu's threshold, written in NumPy.

    For every candidate threshold k the pixels split into class 0 (values <= k) and
    class 1 (values > k). Otsu picks the k that maximises the between-class variance

        sigma_B^2(k) = w0(k) * w1(k) * (mu0(k) - mu1(k))^2

    where w0, w1 are the class probabilities and mu0, mu1 the class means.

    Args:
        image: ``uint8`` (H, W) grayscale image, 0-255.

    Returns:
        Threshold k as an int in [0, 255]. Pixels <= k are the dark class. For an image
        with a single gray level, that level is returned.
    """
    _check_image(image)
    hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    levels = np.arange(256, dtype=np.float64)

    w0 = np.cumsum(hist)                      # pixels with value <= k
    w1 = hist.sum() - w0                      # pixels with value > k
    sum0 = np.cumsum(hist * levels)
    sum1 = sum0[-1] - sum0

    valid = (w0 > 0) & (w1 > 0)
    if not valid.any():
        return int(image.flat[0])

    mu0 = np.divide(sum0, w0, out=np.zeros(256), where=valid)
    mu1 = np.divide(sum1, w1, out=np.zeros(256), where=valid)
    between = np.where(valid, w0 * w1 * (mu0 - mu1) ** 2, -1.0)
    return int(np.argmax(between))


def threshold_mask(image: np.ndarray, t: int, dark_objects: bool = True) -> np.ndarray:
    """Binary mask from a global threshold.

    Args:
        image: ``uint8`` (H, W) image, 0-255.
        t: Threshold in [0, 255].
        dark_objects: If True (EL defects), True where pixel <= t; else True where > t.

    Returns:
        ``bool`` (H, W) mask, True = defect candidate.
    """
    _check_image(image)
    if not 0 <= int(t) <= 255:
        raise ValueError("t must be in [0, 255]")
    return image <= t if dark_objects else image > t


def _box_mean_replicate(image: np.ndarray, block_size: int) -> np.ndarray:
    """Local mean over a block_size x block_size window, replicate border, rounded.

    Uses an integral image (summed-area table), so the cost does not depend on the block
    size. Rounding is half-to-even, the same rule OpenCV uses when it stores the mean.
    """
    pad = block_size // 2
    padded = np.pad(image.astype(np.float64), pad, mode="edge")
    integral = np.pad(padded.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    h, w = image.shape
    b = block_size
    window_sum = (
        integral[b:b + h, b:b + w]
        - integral[0:h, b:b + w]
        - integral[b:b + h, 0:w]
        + integral[0:h, 0:w]
    )
    return np.rint(window_sum / (b * b))


def adaptive_threshold(
    image: np.ndarray, block_size: int, c: int, dark_objects: bool = True
) -> np.ndarray:
    """Local-mean adaptive threshold, written in NumPy.

    Each pixel is compared with the mean of its block_size x block_size neighbourhood
    (replicate border). With ``dark_objects`` it is a candidate when
    ``pixel <= local_mean - c``, which handles the uneven brightness across an EL cell
    that defeats a single global threshold.

    Args:
        image: ``uint8`` (H, W) image, 0-255.
        block_size: Odd window size >= 3.
        c: Integer offset subtracted from the local mean.
        dark_objects: If False, candidates are pixels > local_mean - c instead.

    Returns:
        ``bool`` (H, W) mask, True = defect candidate. Matches
        ``cv2.adaptiveThreshold(..., ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, ...)``.
    """
    _check_image(image)
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer >= 3")
    if int(c) != c:
        raise ValueError("c must be an integer")
    mean = _box_mean_replicate(image, block_size)
    dark = image.astype(np.float64) - mean <= -int(c)
    return dark if dark_objects else ~dark
